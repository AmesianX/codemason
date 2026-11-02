//! Markdown -> ANSI renderer — the terminal-substrate equivalent of the
//! original's Ink markdown renderer, and of the Python port's `rich`-based
//! `render_markdown` (utils/formatting.py `_build_md_renderer`).
//!
//! Substrate divergence (documented): Python rendered through `rich`; Rust has
//! no rich, so this renders pulldown-cmark events directly. The STYLING
//! CONTRACT is carried over exactly — the same absolute-256 palette, for the
//! same reason (named ANSI colours get remapped by the terminal's own palette
//! and collapse under a theme; absolute 256 indices do not):
//!
//!   h1 bold 75 (soft sky blue, left-aligned) · h2 bold 214 (bright orange)
//!   h3 bold 220 (yellow) · h4-h6 bold 245 (grey) · **bold** bold+220
//!   *emph* italic 250 · `code` 180 (warm tan) · links underline 75 / url 240
//!   bullets/numbers bold 75 · blockquote italic 108 · hr 240
//!
//! Code blocks are syntax-highlighted with syntect's Monokai theme (the Python
//! passed code_theme='monokai' to rich, which used Pygments' monokai — same
//! palette family). Output width is min(terminal, 120), like the Python.

use once_cell::sync::Lazy;
use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use syntect::easy::HighlightLines;
use syntect::highlighting::ThemeSet;
use syntect::parsing::SyntaxSet;

const RESET: &str = "\x1b[0m";

fn fg(n: u8) -> String {
    format!("\x1b[38;5;{n}m")
}

struct Styles;
impl Styles {
    fn heading(level: HeadingLevel) -> String {
        let color = match level {
            HeadingLevel::H1 => 75,
            HeadingLevel::H2 => 214,
            HeadingLevel::H3 => 220,
            _ => 245,
        };
        format!("\x1b[1m{}", fg(color))
    }
    fn strong() -> String {
        format!("\x1b[1m{}", fg(220))
    }
    fn emph() -> String {
        format!("\x1b[3m{}", fg(250))
    }
    fn code_inline() -> String {
        fg(180)
    }
    fn link() -> String {
        format!("\x1b[4m{}", fg(75))
    }
    fn link_url() -> String {
        fg(240)
    }
    fn bullet() -> String {
        format!("\x1b[1m{}", fg(75))
    }
    fn quote() -> String {
        format!("\x1b[3m{}", fg(108))
    }
    fn hr() -> String {
        fg(240)
    }
}

static SYNTAXES: Lazy<SyntaxSet> = Lazy::new(SyntaxSet::load_defaults_newlines);
static THEMES: Lazy<ThemeSet> = Lazy::new(ThemeSet::load_defaults);

fn terminal_width() -> usize {
    let cols = crossterm::terminal::size().map(|(c, _)| c as usize).unwrap_or(100);
    cols.min(120)
}

/// Highlight one fenced code block with Monokai; falls back to the plain text
/// (tan) when the language is unknown.
fn highlight_code(code: &str, lang: &str) -> Vec<String> {
    // syntect's default set names it "base16-monokai"-adjacent keys vary by
    // build; fall back through the dark defaults rather than panic.
    let theme = THEMES
        .themes
        .get("Monokai")
        .or_else(|| THEMES.themes.get("base16-monokai.dark"))
        .or_else(|| THEMES.themes.get("base16-eighties.dark"))
        .or_else(|| THEMES.themes.values().next());
    let Some(theme) = theme else {
        return code
            .lines()
            .map(|l| format!("{}{}{}", Styles::code_inline(), l, RESET))
            .collect();
    };
    let syntax = if lang.is_empty() {
        None
    } else {
        SYNTAXES
            .find_syntax_by_token(lang)
            .or_else(|| SYNTAXES.find_syntax_by_extension(lang))
    };
    let Some(syntax) = syntax else {
        return code
            .lines()
            .map(|l| format!("{}{}{}", Styles::code_inline(), l, RESET))
            .collect();
    };
    let mut h = HighlightLines::new(syntax, theme);
    let mut out = Vec::new();
    for line in code.lines() {
        match h.highlight_line(line, &SYNTAXES) {
            Ok(ranges) => {
                let mut styled = String::new();
                for (style, text) in ranges {
                    let c = style.foreground;
                    // Absolute 256 (nearest index), consistent with the rest of
                    // the UI's palette-bypass strategy.
                    let idx = crate::utils::formatting::hex_to_ansi256(&format!(
                        "{:02x}{:02x}{:02x}",
                        c.r, c.g, c.b
                    ));
                    styled.push_str(&format!("\x1b[38;5;{idx}m{text}"));
                }
                styled.push_str(RESET);
                out.push(styled);
            }
            Err(_) => out.push(line.to_string()),
        }
    }
    out
}

#[derive(Default)]
struct ListState {
    ordered: bool,
    index: u64,
}

/// Render markdown to ANSI text. Any parse hiccup falls back to the raw text —
/// one block of UI is never worth taking the program down (same posture as the
/// Python's except-fallback).
pub fn render_markdown_ansi(text: &str) -> String {
    // Tell the panic hook this one is expected and handled. Without the guard
    // the hook would restore the terminal — leave raw mode, re-enable echo —
    // for a panic we are about to swallow, and the engine would carry on
    // believing it still owns a raw terminal. The hook runs before unwinding,
    // so catch_unwind alone does not keep it out.
    let _guard = crate::utils::panic_guard::expected_panic();
    std::panic::catch_unwind(|| render_inner(text)).unwrap_or_else(|_| text.to_string())
}

fn render_inner(text: &str) -> String {
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_TABLES);
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    let parser = Parser::new_ext(text, opts);
    let width = terminal_width();

    let mut out = String::new();
    // Inline style stack: what to re-apply after RESET.
    let mut style_stack: Vec<String> = Vec::new();
    let mut line = String::new(); // current inline accumulation (with ANSI)
    let mut line_width = 0usize;
    let mut lists: Vec<ListState> = Vec::new();
    let mut quote_depth = 0usize;
    let mut in_code_block = false;
    let mut code_lang = String::new();
    let mut code_buf = String::new();
    let mut link_url = String::new();
    let mut in_table = false;
    let mut table_rows: Vec<Vec<String>> = Vec::new();
    let mut table_cur_row: Vec<String> = Vec::new();
    let mut table_cell = String::new();
    let mut in_heading: Option<HeadingLevel> = None;

    let indent_of = |lists: &[ListState], quote_depth: usize| -> String {
        let mut s = String::new();
        for _ in 0..quote_depth {
            s.push_str(&format!("{}▌ {}", Styles::quote(), RESET));
        }
        if !lists.is_empty() {
            s.push_str(&"  ".repeat(lists.len() - 1));
        }
        s
    };

    macro_rules! flush_line {
        ($out:ident, $line:ident, $line_width:ident) => {
            if !$line.is_empty() {
                $out.push_str(&$line);
                $out.push('\n');
                $line = String::new();
                $line_width = 0;
            }
        };
    }

    let current_style =
        |style_stack: &[String]| -> String { style_stack.concat() };

    let mut push_text = |line: &mut String,
                         line_width: &mut usize,
                         out: &mut String,
                         style_stack: &[String],
                         lists: &[ListState],
                         quote_depth: usize,
                         text: &str| {
        // Word-wrap at the render width, preserving the active style across
        // wrapped lines.
        for word in text.split_inclusive(' ') {
            let w = crate::utils::formatting::display_width(word.trim_end());
            if *line_width + w > width && *line_width > 0 {
                out.push_str(line);
                out.push('\n');
                line.clear();
                let ind = indent_of(lists, quote_depth);
                *line_width = crate::utils::formatting::display_width(
                    &strip_ansi(&ind),
                );
                line.push_str(&ind);
                line.push_str(&current_style(style_stack));
            }
            line.push_str(word);
            *line_width += crate::utils::formatting::display_width(word);
        }
    };

    for event in parser {
        match event {
            Event::Start(tag) => match tag {
                Tag::Heading { level, .. } => {
                    flush_line!(out, line, line_width);
                    if !out.is_empty() && !out.ends_with("\n\n") {
                        out.push('\n');
                    }
                    in_heading = Some(level);
                    let style = Styles::heading(level);
                    line.push_str(&style);
                    style_stack.push(style);
                }
                Tag::Paragraph => {
                    if !out.is_empty() && !out.ends_with("\n\n") && line.is_empty() && lists.is_empty() {
                        out.push('\n');
                    }
                    let ind = indent_of(&lists, quote_depth);
                    if line.is_empty() && !ind.is_empty() && lists.is_empty() {
                        line.push_str(&ind);
                        line_width = crate::utils::formatting::display_width(&strip_ansi(&ind));
                    }
                }
                Tag::BlockQuote(_) => {
                    flush_line!(out, line, line_width);
                    quote_depth += 1;
                }
                Tag::CodeBlock(kind) => {
                    flush_line!(out, line, line_width);
                    if !out.is_empty() && !out.ends_with("\n\n") {
                        out.push('\n');
                    }
                    in_code_block = true;
                    code_lang = match kind {
                        CodeBlockKind::Fenced(l) => l.to_string(),
                        CodeBlockKind::Indented => String::new(),
                    };
                    code_buf.clear();
                }
                Tag::List(start) => {
                    flush_line!(out, line, line_width);
                    lists.push(ListState {
                        ordered: start.is_some(),
                        index: start.unwrap_or(1),
                    });
                }
                Tag::Item => {
                    flush_line!(out, line, line_width);
                    let ind = indent_of(&lists, quote_depth);
                    let marker = if let Some(l) = lists.last_mut() {
                        if l.ordered {
                            let m = format!("{}{}. {}", Styles::bullet(), l.index, RESET);
                            l.index += 1;
                            m
                        } else {
                            format!("{}• {}", Styles::bullet(), RESET)
                        }
                    } else {
                        String::new()
                    };
                    line.push_str(&ind);
                    line.push_str(&marker);
                    line_width = crate::utils::formatting::display_width(&strip_ansi(&line));
                }
                Tag::Emphasis => {
                    let s = Styles::emph();
                    line.push_str(&s);
                    style_stack.push(s);
                }
                Tag::Strong => {
                    let s = Styles::strong();
                    line.push_str(&s);
                    style_stack.push(s);
                }
                Tag::Strikethrough => {
                    let s = "\x1b[9m".to_string();
                    line.push_str(&s);
                    style_stack.push(s);
                }
                Tag::Link { dest_url, .. } => {
                    link_url = dest_url.to_string();
                    let s = Styles::link();
                    line.push_str(&s);
                    style_stack.push(s);
                }
                Tag::Table(_) => {
                    flush_line!(out, line, line_width);
                    in_table = true;
                    table_rows.clear();
                }
                Tag::TableHead | Tag::TableRow => {
                    table_cur_row = Vec::new();
                }
                Tag::TableCell => {
                    table_cell = String::new();
                }
                _ => {}
            },
            Event::End(tag) => match tag {
                TagEnd::Heading(_) => {
                    style_stack.pop();
                    line.push_str(RESET);
                    flush_line!(out, line, line_width);
                    in_heading = None;
                }
                TagEnd::Paragraph => {
                    flush_line!(out, line, line_width);
                }
                TagEnd::BlockQuote(_) => {
                    flush_line!(out, line, line_width);
                    quote_depth = quote_depth.saturating_sub(1);
                }
                TagEnd::CodeBlock => {
                    in_code_block = false;
                    let code = code_buf.trim_end_matches('\n');
                    for hl in highlight_code(code, &code_lang) {
                        out.push_str("    ");
                        out.push_str(&hl);
                        out.push('\n');
                    }
                    code_buf.clear();
                }
                TagEnd::List(_) => {
                    lists.pop();
                    flush_line!(out, line, line_width);
                }
                TagEnd::Item => {
                    flush_line!(out, line, line_width);
                }
                TagEnd::Emphasis | TagEnd::Strong | TagEnd::Strikethrough => {
                    style_stack.pop();
                    line.push_str(RESET);
                    line.push_str(&current_style(&style_stack));
                }
                TagEnd::Link => {
                    style_stack.pop();
                    line.push_str(RESET);
                    if !link_url.is_empty() && !link_url.starts_with('#') {
                        line.push_str(&format!(
                            " {}({}){}",
                            Styles::link_url(),
                            link_url,
                            RESET
                        ));
                        line_width += crate::utils::formatting::display_width(&link_url) + 3;
                    }
                    line.push_str(&current_style(&style_stack));
                    link_url.clear();
                }
                TagEnd::TableCell => {
                    table_cur_row.push(table_cell.clone());
                }
                TagEnd::TableHead | TagEnd::TableRow => {
                    table_rows.push(table_cur_row.clone());
                }
                TagEnd::Table => {
                    in_table = false;
                    out.push_str(&render_table(&table_rows));
                }
                _ => {}
            },
            Event::Text(t) => {
                if in_code_block {
                    code_buf.push_str(&t);
                } else if in_table {
                    table_cell.push_str(&t);
                } else if in_heading.is_some() {
                    line.push_str(&t);
                    line_width += crate::utils::formatting::display_width(&t);
                } else {
                    push_text(
                        &mut line,
                        &mut line_width,
                        &mut out,
                        &style_stack,
                        &lists,
                        quote_depth,
                        &t,
                    );
                }
            }
            Event::Code(t) => {
                let styled = format!("{}{}{}{}", Styles::code_inline(), t, RESET, current_style(&style_stack));
                if in_table {
                    table_cell.push_str(&t);
                } else {
                    line.push_str(&styled);
                    line_width += crate::utils::formatting::display_width(&t);
                }
            }
            Event::SoftBreak => {
                if in_table {
                    table_cell.push(' ');
                } else {
                    push_text(
                        &mut line,
                        &mut line_width,
                        &mut out,
                        &style_stack,
                        &lists,
                        quote_depth,
                        " ",
                    );
                }
            }
            Event::HardBreak => {
                flush_line!(out, line, line_width);
            }
            Event::Rule => {
                flush_line!(out, line, line_width);
                out.push_str(&format!(
                    "{}{}{}\n",
                    Styles::hr(),
                    "─".repeat(width.min(80)),
                    RESET
                ));
            }
            Event::TaskListMarker(done) => {
                line.push_str(if done { "[x] " } else { "[ ] " });
                line_width += 4;
            }
            Event::Html(t) | Event::InlineHtml(t) => {
                if in_code_block {
                    code_buf.push_str(&t);
                } else {
                    push_text(
                        &mut line,
                        &mut line_width,
                        &mut out,
                        &style_stack,
                        &lists,
                        quote_depth,
                        &t,
                    );
                }
            }
            _ => {}
        }
    }
    flush_line!(out, line, line_width);
    let _ = line_width;
    out.trim_end_matches('\n').to_string()
}

fn strip_ansi(s: &str) -> String {
    static RE: Lazy<regex::Regex> =
        Lazy::new(|| regex::Regex::new(r"\x1b\[[0-9;]*m").unwrap());
    RE.replace_all(s, "").into_owned()
}

/// Box-drawn table with display_width-aware column sizing (rich table
/// equivalent, simplified to the heavy-head style rich defaults to).
fn render_table(rows: &[Vec<String>]) -> String {
    if rows.is_empty() {
        return String::new();
    }
    let cols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    let mut widths = vec![0usize; cols];
    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            widths[i] = widths[i].max(crate::utils::formatting::display_width(cell));
        }
    }
    let mut out = String::new();
    let border = fg(240);
    let sep_line = |l: &str, m: &str, r: &str| -> String {
        let mut s = format!("{border}{l}");
        for (i, w) in widths.iter().enumerate() {
            s.push_str(&"─".repeat(w + 2));
            s.push_str(if i + 1 < cols { m } else { r });
        }
        s.push_str(RESET);
        s.push('\n');
        s
    };
    out.push_str(&sep_line("┌", "┬", "┐"));
    for (ri, row) in rows.iter().enumerate() {
        out.push_str(&format!("{border}│{RESET}"));
        for (i, w) in widths.iter().enumerate() {
            let cell = row.get(i).map(|s| s.as_str()).unwrap_or("");
            let pad = w - crate::utils::formatting::display_width(cell);
            if ri == 0 {
                out.push_str(&format!(" \x1b[1m{cell}\x1b[0m{} ", " ".repeat(pad)));
            } else {
                out.push_str(&format!(" {cell}{} ", " ".repeat(pad)));
            }
            out.push_str(&format!("{border}│{RESET}"));
        }
        out.push('\n');
        if ri == 0 && rows.len() > 1 {
            out.push_str(&sep_line("├", "┼", "┤"));
        }
    }
    out.push_str(&sep_line("└", "┴", "┘"));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_heading_with_256_color() {
        let out = render_inner("# Title");
        assert!(out.contains("\x1b[38;5;75m"));
        assert!(out.contains("Title"));
    }

    #[test]
    fn renders_bold_yellow() {
        let out = render_inner("some **bold** text");
        assert!(out.contains("\x1b[38;5;220m"));
    }

    #[test]
    fn code_block_highlighted() {
        let out = render_inner("```python\nprint('hi')\n```");
        assert!(out.contains("print"));
        assert!(out.contains("\x1b[38;5;"));
    }

    #[test]
    fn table_draws_grid() {
        let out = render_inner("|a|b|\n|-|-|\n|1|2|");
        assert!(out.contains('┌') && out.contains('┘'));
    }
}
