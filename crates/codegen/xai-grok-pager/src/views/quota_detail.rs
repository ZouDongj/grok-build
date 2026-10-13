//! Quota detail modal: the coding-plan windows' remaining amounts, next
//! reset times and available reset counts. Opened by clicking the status
//! chip (or when the chip is hovered + Enter); Esc / clicking the chip again
//! closes it.

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Paragraph};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;

use crate::theme::Theme;
use crate::views::agent_status::QuotaChipState;

fn format_count(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn until(ms: u64) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    if ms <= now {
        return "即将重置".into();
    }
    let secs = (ms - now) / 1000;
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    if h >= 24 {
        format!("{}天后", h / 24)
    } else if h > 0 {
        format!("{h}小时{m}分后")
    } else {
        format!("{m}分钟后")
    }
}

fn window_rows(
    out: &mut Vec<Line<'static>>,
    label: &str,
    w: &crate::views::agent_status::QuotaWindow,
    theme: &Theme,
) {
    let remaining = w.total.saturating_sub(w.used);
    let rem_pct = 100 - w.used_pct;
    let title = Style::default().fg(theme.text_primary);
    let dim = Style::default().fg(theme.gray);
    let value = Style::default().fg(theme.text_primary).add_modifier(Modifier::BOLD);
    out.push(Line::from(vec![
        Span::styled(format!("  {label}  "), dim),
        Span::styled(
            format!("剩余 {} / {}", format_count(remaining), format_count(w.total)),
            value,
        ),
        Span::styled(format!("（{rem_pct}%）"), dim),
    ]));
    if let Some(next) = w.next_reset_ms {
        out.push(Line::from(vec![
            Span::styled("         ", dim),
            Span::styled(format!("重置于 {}", until(next)), dim),
        ]));
    }
}

/// Render the modal centered in `area`; returns the modal's rect for
/// occlusion tracking.
pub fn render_quota_detail(
    buf: &mut Buffer,
    area: Rect,
    q: &QuotaChipState,
    theme: &Theme,
) -> Rect {
    let mut lines: Vec<Line<'static>> = Vec::new();
    let head = Style::default()
        .fg(theme.accent_plan)
        .add_modifier(Modifier::BOLD);
    lines.push(Line::from(Span::styled("套餐额度（GLM Coding Plan）", head)));
    lines.push(Line::from(""));
    if let Some(w) = q.five_hour.as_ref() {
        window_rows(&mut lines, "5 小时窗口", w, theme);
    }
    if let Some(w) = q.week.as_ref() {
        window_rows(&mut lines, "7 天窗口", w, theme);
    }
    lines.push(Line::from(""));
    let (f, w) = (q.reset_five.unwrap_or(0), q.reset_week.unwrap_or(0));
    lines.push(Line::from(vec![
        Span::styled("  可用重置卡  ", Style::default().fg(theme.gray)),
        Span::styled(format!("5 小时 ×{f}，7 天 ×{w}"), Style::default().fg(theme.text_primary)),
    ]));
    lines.push(Line::from(Span::styled(
        "  使用：/quota reset five_hour 或 /quota reset week",
        Style::default().fg(theme.gray),
    )));
    lines.push(Line::from(Span::styled(
        "  Esc 关闭",
        Style::default().fg(theme.gray_dim),
    )));

    let width = 52u16.min(area.width.saturating_sub(4));
    let height = (lines.len() as u16 + 2).min(area.height.saturating_sub(2));
    let x = area.x + (area.width.saturating_sub(width)) / 2;
    let y = area.y + (area.height.saturating_sub(height)) / 2;
    let rect = Rect { x, y, width, height };
    Clear.render(rect, buf);
    let block = ratatui::widgets::Block::bordered()
        .border_style(Style::default().fg(theme.gray))
        .title(Span::styled(" Quota ", Style::default().fg(theme.accent_plan)));
    use ratatui::widgets::Widget;
    Paragraph::new(lines)
        .block(block)
        .render(rect, buf);
    rect
}
