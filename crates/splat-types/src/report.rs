use serde::Serialize;

/// Anything that produces a metrics report.
///
/// Every metric-producing component returns a plain struct implementing this
/// trait; the CLI writes it to `reports/*.json`. Implementors only need
/// `#[derive(Serialize)]` and an empty `impl Report for X {}`.
pub trait Report: Serialize {
    /// Pretty-printed JSON with a trailing newline (stable field order, so
    /// reports diff cleanly).
    fn to_json(&self) -> Result<String, serde_json::Error> {
        let mut s = serde_json::to_string_pretty(self)?;
        s.push('\n');
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize)]
    struct R {
        a: u32,
    }
    impl Report for R {}

    #[test]
    fn json_has_newline() {
        assert_eq!(R { a: 1 }.to_json().unwrap(), "{\n  \"a\": 1\n}\n");
    }
}
