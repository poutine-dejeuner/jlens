/**
 * slurm-watcher.ts — Pi extension for SLURM job monitoring
 *
 * Detects `sbatch` calls, extracts job IDs, polls `squeue`, tails logs,
 * and reports completion/failure. Keeps the agent in the loop until
 * every job finishes.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";

// ── per-session state ──
interface TrackedJob {
  jobId: string;
  startedAt: number;
  outputFile?: string;
  errorFile?: string;
  lastPollAt: number;
  done: boolean;
  exitCode?: number;
}

const activeJobs = new Map<string, TrackedJob>();

// ── helpers ──

function extractJobId(stdout: string): string | null {
  const m = stdout.match(/Submitted batch job (\d+)/);
  return m ? m[1] : null;
}

async function squeueJob(jobId: string): Promise<{
  found: boolean;
  state?: string;
  node?: string;
}> {
  return new Promise((resolve) => {
    const proc = spawn("squeue", ["-j", jobId, "-h", "-o", "%T|%N"], {
      timeout: 10_000,
    });
    let out = "";
    proc.stdout?.on("data", (d: Buffer) => (out += d.toString()));
    proc.on("close", (code) => {
      if (code !== 0 || !out.trim()) return resolve({ found: false });
      const parts = out.trim().split("|");
      resolve({ found: true, state: parts[0]?.trim(), node: parts[1]?.trim() });
    });
    proc.on("error", () => resolve({ found: false }));
  });
}

function tailFile(path: string, lines = 15): string {
  try {
    const data = readFileSync(path, "utf-8");
    return data.split("\n").slice(-lines).join("\n");
  } catch {
    return "(file not available yet)";
  }
}

function elapsed(ms: number): string {
  const s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h > 0 ? `${h}h${m}m` : `${m}m${s % 60}s`;
}

// ── optional log path extraction from sbatch args ──
function extractLogPaths(command: string): {
  output?: string;
  error?: string;
} {
  const outM = command.match(/--output[= ](\S+)/);
  const errM = command.match(/--error[= ](\S+)/);
  return { output: outM?.[1], error: errM?.[1] };
}

// ── main watcher loop ──
async function watchJob(
  job: TrackedJob,
  pi: ExtensionAPI,
) {
  const pollInterval = 30_000; // 30s
  const maxPolls = 720; // ~6h at 30s

  for (let i = 0; i < maxPolls; i++) {
    await new Promise((r) => setTimeout(r, pollInterval));
    if (job.done) return;

    const status = await squeueJob(job.jobId);
    job.lastPollAt = Date.now();

    if (!status.found) {
      // Job finished — squeue no longer shows it
      job.done = true;
      pi.sendUserMessage(
        `🔔 SLURM job **${job.jobId}** has finished (elapsed: ${elapsed(Date.now() - job.startedAt)}).\n\n` +
          `Please check the logs for results and report progress.`,
        { deliverAs: "followUp" },
      );
      return;
    }

    // Still running — occasional progress update
    if (i % 10 === 0 && i > 0) {
      const tail = job.outputFile ? tailFile(job.outputFile, 8) : "";
      pi.sendUserMessage(
        `⏳ SLURM job **${job.jobId}** still running (${elapsed(Date.now() - job.startedAt)}, state: ${status.state ?? "?"}, node: ${status.node ?? "?"}).\n` +
          (tail ? `\nLatest output:\n\`\`\`\n${tail}\n\`\`\`` : ""),
        { deliverAs: "nextTurn" },
      );
    }

    // Detect failure states
    if (status.state && ["FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL"].includes(status.state)) {
      job.done = true;
      job.exitCode = 1;
      const errTail = job.errorFile ? tailFile(job.errorFile, 20) : "";
      pi.sendUserMessage(
        `🚨 SLURM job **${job.jobId}** FAILED (state: ${status.state}).\n\n` +
          (errTail ? `Error log:\n\`\`\`\n${errTail}\n\`\`\`` : "No error log available.") +
          `\n\nPlease diagnose the error and propose a fix.`,
        { deliverAs: "followUp" },
      );
      return;
    }
  }

  // Timeout
  pi.sendUserMessage(
    `⏰ SLURM job **${job.jobId}** monitoring timed out after ${elapsed(Date.now() - job.startedAt)}. It may still be running.`,
    { deliverAs: "followUp" },
  );
}

// ── extension ──

export default function (pi: ExtensionAPI) {
  // Track sbatch calls on tool_result (after execution)
  pi.on("tool_result", (event, ctx) => {
    if (event.toolName !== "bash" || !event.input?.command) return;

    const cmd: string = event.input.command;

    // Only care about sbatch
    if (!cmd.includes("sbatch")) return;

    const jobId = extractJobId(event.content?.[0]?.text ?? "");
    if (!jobId) return;

    // Skip if we already have a watcher for this job (e.g., compaction replay)
    if (activeJobs.has(jobId)) return;

    // Check if the job actually exists in squeue right now
    // During compaction replay, old jobs won't be in squeue.
    // Quick synchronous check via spawnSync equivalent — skip if
    // job doesn't exist (it's a replay of an old sbatch).
    const { spawnSync } = require("node:child_process");
    const checkProc = spawnSync("squeue", ["-j", jobId, "-h", "-o", "%T"], {
      timeout: 5000,
    });
    if (checkProc.status !== 0 || !checkProc.stdout.toString().trim()) {
      // Job doesn't exist in squeue — likely a compaction replay, skip
      return;
    }

    const paths = extractLogPaths(cmd);

    const job: TrackedJob = {
      jobId,
      startedAt: Date.now(),
      outputFile: paths.output?.replace(
        "%j",
        jobId,
      ).replace("%a", "*"),
      errorFile: paths.error?.replace(
        "%j",
        jobId,
      ).replace("%a", "*"),
      lastPollAt: Date.now(),
      done: false,
    };

    activeJobs.set(jobId, job);

    // Start async watcher (fire-and-forget)
    watchJob(job, pi).catch((err) => {
      console.error(`[slurm-watcher] watchJob failed:`, err);
    });

    // Return a note that monitoring is active
    return {
      details: {
        ...(event.details ?? {}),
        slurmWatcher: `Monitoring job ${jobId}. Will report when done.`,
      },
    };
  });

  // Register a command for manual monitoring
  pi.registerCommand("slurm-status", {
    description: "Show status of tracked SLURM jobs",
    handler: async (_args, ctx) => {
      if (activeJobs.size === 0) {
        ctx.ui.notify("No SLURM jobs tracked in this session.", "info");
        return;
      }
      const lines: string[] = [];
      for (const [id, job] of activeJobs) {
        const status = await squeueJob(id);
        const state = status.found
          ? `${status.state} @ ${status.node}`
          : "done (no longer in queue)";
        lines.push(
          `  ${id}: ${state} (${elapsed(Date.now() - job.startedAt)})`,
        );
      }
      ctx.ui.notify(
        `Tracked SLURM jobs:\n${lines.join("\n")}`,
        "info",
      );
    },
  });

  // Register a command to clear stale watchers
  pi.registerCommand("slurm-clear", {
    description: "Clear all tracked SLURM job watchers",
    handler: async (_args, ctx) => {
      const count = activeJobs.size;
      if (count === 0) {
        ctx.ui.notify("No SLURM jobs tracked.", "info");
        return;
      }
      const ids = [...activeJobs.keys()];
      activeJobs.clear();
      ctx.ui.notify(
        `Cleared ${count} SLURM job watcher(s): ${ids.join(", ")}`,
        "info",
      );
    },
  });

  // Cleanup on session shutdown
  pi.on("session_shutdown", () => {
    activeJobs.clear();
  });
}
