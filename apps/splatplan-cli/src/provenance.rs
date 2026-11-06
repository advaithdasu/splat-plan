//! Where a number came from: command, commit and machine.

use serde::{Deserialize, Serialize};

/// Provenance stamped into every report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    /// The command that produced the report.
    pub command: String,
    /// `git rev-parse --short HEAD`, or "unknown".
    pub commit: String,
    /// True if the working tree had uncommitted changes.
    pub dirty: bool,
    /// CPU model (and core counts on macOS), for performance numbers.
    pub machine: String,
    /// Target OS and architecture.
    pub target: String,
}

/// Uncommitted changes outside `reports/` and `README.md`. Generated reports
/// and prose don't change results, so they don't make a tree look dirty.
fn dirty_tree() -> bool {
    run("git", &DIRTY_ARGS).is_some_and(|s| !s.is_empty())
}

const DIRTY_ARGS: [&str; 6] = [
    "status",
    "--porcelain",
    "--",
    ".",
    ":(exclude)reports",
    ":(exclude)README.md",
];

fn run(cmd: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(cmd).args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// CPU description: on macOS the chip name plus performance/efficiency core
/// counts (Apple Silicon has both), elsewhere the model name from /proc/cpuinfo.
pub fn machine() -> String {
    if cfg!(target_os = "macos") {
        let chip = run("sysctl", &["-n", "machdep.cpu.brand_string"]).unwrap_or_default();
        let p = run("sysctl", &["-n", "hw.perflevel0.physicalcpu"]).unwrap_or_default();
        let e = run("sysctl", &["-n", "hw.perflevel1.physicalcpu"]).unwrap_or_default();
        format!("{chip} ({p}P + {e}E cores)")
    } else {
        let cpu = std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("model name"))
                    .and_then(|l| l.split(':').nth(1))
                    .map(|v| v.trim().to_string())
            })
            .unwrap_or_else(|| "unknown CPU".into());
        let n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0);
        format!("{cpu} ({n} threads)")
    }
}

impl Provenance {
    /// Captures the current commit and machine for `command`.
    pub fn capture(command: &str) -> Self {
        Provenance {
            command: command.to_string(),
            commit: run("git", &["rev-parse", "--short", "HEAD"])
                .unwrap_or_else(|| "unknown".into()),
            dirty: dirty_tree(),
            machine: machine(),
            target: format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A rejected pathspec would make `run` return `None`, which reads as
    /// "clean": the exact command must succeed.
    #[test]
    fn dirty_check_command_is_accepted_by_git() {
        if run("git", &["rev-parse", "--git-dir"]).is_some() {
            assert!(run("git", &DIRTY_ARGS).is_some());
        }
    }
}
