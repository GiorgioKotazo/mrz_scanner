//! Shared subprocess helpers, not a standalone test target.
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

pub fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

pub fn fixture(name: &str) -> PathBuf {
    root().join("tests/regression/fixtures").join(name)
}

pub fn cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_mrz_scanner"))
        .current_dir(root())
        .args(args)
        .output()
        .expect("run CLI")
}

pub fn assert_no_candidate(output: &Output, code: i32) {
    assert_eq!(output.status.code(), Some(code), "{output:?}");
    assert!(!String::from_utf8_lossy(&output.stdout).contains("MRZ (OCR):"));
}
