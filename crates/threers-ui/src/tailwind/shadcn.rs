//! shadcn/ui's look: its neutral theme and its components as classes.

use super::theme::{parse_color, Paint, Tailwind};

/// shadcn/ui's neutral theme (its `:root` and `.dark` variables): name,
/// light, dark.
const COLORS: [(&str, &str, &str); 18] = [
    ("background", "oklch(1 0 0)", "oklch(0.145 0 0)"),
    ("foreground", "oklch(0.145 0 0)", "oklch(0.985 0 0)"),
    ("card", "oklch(1 0 0)", "oklch(0.205 0 0)"),
    ("card-foreground", "oklch(0.145 0 0)", "oklch(0.985 0 0)"),
    ("popover", "oklch(1 0 0)", "oklch(0.205 0 0)"),
    ("popover-foreground", "oklch(0.145 0 0)", "oklch(0.985 0 0)"),
    ("primary", "oklch(0.205 0 0)", "oklch(0.922 0 0)"),
    ("primary-foreground", "oklch(0.985 0 0)", "oklch(0.205 0 0)"),
    ("secondary", "oklch(0.97 0 0)", "oklch(0.269 0 0)"),
    (
        "secondary-foreground",
        "oklch(0.205 0 0)",
        "oklch(0.985 0 0)",
    ),
    ("muted", "oklch(0.97 0 0)", "oklch(0.269 0 0)"),
    ("muted-foreground", "oklch(0.556 0 0)", "oklch(0.708 0 0)"),
    ("accent", "oklch(0.97 0 0)", "oklch(0.269 0 0)"),
    ("accent-foreground", "oklch(0.205 0 0)", "oklch(0.985 0 0)"),
    (
        "destructive",
        "oklch(0.577 0.245 27.325)",
        "oklch(0.704 0.191 22.216)",
    ),
    ("border", "oklch(0.922 0 0)", "oklch(1 0 0 / 10%)"),
    ("input", "oklch(0.922 0 0)", "oklch(1 0 0 / 15%)"),
    ("ring", "oklch(0.708 0 0)", "oklch(0.556 0 0)"),
];

/// Its components, as their recipes give them (what this UI has no
/// property for, such as `aria-*` and SVG sizing, left out). A variant or
/// size goes after the base class: `button button-outline button-sm`.
const COMPONENTS: &[(&str, &str)] = &[
    ("page", "bg-background text-foreground"),
    (
        "button",
        "inline-flex shrink-0 items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium \
         transition-all outline-none select-none h-9 px-4 py-2 bg-primary text-primary-foreground shadow-xs \
         hover:bg-primary/90 focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 \
         disabled:pointer-events-none disabled:opacity-50",
    ),
    (
        "button-destructive",
        "bg-destructive text-white shadow-xs hover:bg-destructive/90 focus-visible:ring-destructive/20 \
         dark:focus-visible:ring-destructive/40 dark:bg-destructive/60",
    ),
    (
        "button-outline",
        "border bg-background text-foreground shadow-xs hover:bg-accent hover:text-accent-foreground \
         dark:bg-input/30 dark:border-input dark:hover:bg-input/50",
    ),
    ("button-secondary", "bg-secondary text-secondary-foreground shadow-xs hover:bg-secondary/80"),
    ("button-ghost", "bg-transparent text-foreground shadow-none hover:bg-accent hover:text-accent-foreground dark:hover:bg-accent/50"),
    ("button-link", "bg-transparent text-primary shadow-none hover:bg-transparent hover:underline"),
    ("button-sm", "h-8 gap-1.5 px-3 py-0"),
    ("button-lg", "h-10 px-6 py-0"),
    ("button-icon", "size-9 p-0"),
    ("card", "flex flex-col gap-6 rounded-xl border bg-card py-6 text-card-foreground shadow-sm"),
    ("card-header", "flex flex-col gap-1.5 px-6"),
    ("card-title", "font-semibold leading-none"),
    ("card-description", "text-sm text-muted-foreground"),
    ("card-content", "px-6"),
    ("card-footer", "flex items-center px-6"),
    (
        "input",
        "flex h-9 w-full min-w-0 rounded-md border border-input bg-transparent px-3 py-1 text-base shadow-xs \
         transition-all outline-none md:text-sm dark:bg-input/30 focus-visible:border-ring focus-visible:ring-[3px] \
         focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50",
    ),
    (
        "textarea",
        "flex min-h-16 w-full rounded-md border border-input bg-transparent px-3 py-2 text-base shadow-xs \
         transition-all outline-none md:text-sm dark:bg-input/30 focus-visible:border-ring focus-visible:ring-[3px] \
         focus-visible:ring-ring/50 disabled:opacity-50",
    ),
    ("label", "flex items-center gap-2 text-sm font-medium leading-none select-none"),
    (
        "badge",
        "inline-flex w-fit shrink-0 items-center justify-center gap-1 overflow-hidden whitespace-nowrap rounded-md \
         border border-transparent bg-primary px-2 py-0.5 text-xs font-medium text-primary-foreground transition-all",
    ),
    ("badge-secondary", "border-transparent bg-secondary text-secondary-foreground"),
    ("badge-destructive", "border-transparent bg-destructive text-white dark:bg-destructive/60"),
    ("badge-outline", "border-border bg-transparent text-foreground"),
    ("separator", "h-px w-full shrink-0 bg-border"),
    ("separator-vertical", "h-full w-px shrink-0 bg-border"),
    ("skeleton", "rounded-md bg-accent"),
    ("alert", "relative flex w-full flex-col gap-0.5 rounded-lg border bg-card px-4 py-3 text-sm text-card-foreground"),
    ("alert-destructive", "text-destructive"),
    ("alert-title", "font-medium leading-none"),
    ("alert-description", "text-sm text-muted-foreground"),
    ("tabs-list", "inline-flex h-9 w-fit items-center justify-center rounded-lg bg-muted p-[3px] text-muted-foreground"),
    (
        "tabs-trigger",
        "inline-flex h-full flex-1 items-center justify-center gap-1.5 rounded-md border border-transparent px-2 py-1 \
         text-sm font-medium whitespace-nowrap text-foreground transition-all focus-visible:border-ring \
         focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50 \
         dark:text-muted-foreground",
    ),
    ("tabs-trigger-active", "bg-background shadow-sm dark:border-input dark:bg-input/30 dark:text-foreground"),
    ("tooltip", "w-fit rounded-md bg-foreground px-3 py-1.5 text-xs text-background"),
    ("popover", "w-72 rounded-md border bg-popover p-4 text-popover-foreground shadow-md"),
    ("dialog", "flex w-full max-w-lg flex-col gap-4 rounded-lg border bg-background p-6 shadow-lg"),
    ("progress", "relative h-2 w-full overflow-hidden rounded-full bg-primary/20"),
    ("progress-indicator", "h-full bg-primary transition-all"),
];

impl Tailwind {
    /// shadcn/ui's look over Tailwind's defaults: the neutral theme's
    /// colours, each switching with the appearance (`bg-background`,
    /// `text-foreground`, `bg-primary`, `text-primary-foreground`,
    /// `secondary`, `muted`, `accent`, `destructive`, `card`, `popover`,
    /// `border`, `input`, `ring`); its radius (`rounded-lg` 10px, `md` 8,
    /// `sm` 6, `xl` 14); borders in `border` and rings in `ring`; and its
    /// components as classes:
    ///
    /// - `button`, with `button-destructive`, `-outline`, `-secondary`,
    ///   `-ghost` or `-link`, and `button-sm`, `-lg` or `-icon`;
    /// - `card` with `card-header`, `-title`, `-description`, `-content`
    ///   and `-footer`; `alert` (`alert-destructive`, `-title`,
    ///   `-description`); `dialog`, `popover`, `tooltip`;
    /// - `input`, `textarea`, `label`; `badge` (`-secondary`,
    ///   `-destructive`, `-outline`); `separator` (`-vertical`), `skeleton`,
    ///   `progress` and `progress-indicator`; `tabs-list`, `tabs-trigger`
    ///   and `tabs-trigger-active`; `page` for the window's background.
    ///
    /// Each is a [component](Self::component): classes after it go over it
    /// (`button w-full`), and the builder can change any of them.
    ///
    /// ```
    /// use threers_ui::prelude::*;
    /// use threers_ui::tailwind::Tailwind;
    ///
    /// let mut ui = Ui::new();
    /// ui.set_global(Tailwind::shadcn());
    /// let root = ui.root();
    /// let card = ui.add_tw(root, "card w-80");
    /// let header = ui.add_tw(card, "card-header");
    /// let title = ui.add_tw(header, "card-title");
    /// ui.add_text(title, "Create project");
    /// let save = ui.add_tw(card, "button mx-6");
    /// ui.add_text(save, "Deploy");
    /// let light = ui.style(save).background;
    /// ui.set_theme(Theme::Dark);
    /// ui.step(1.0); // `transition-all`: the colours move there
    /// assert_ne!(ui.style(save).background, light, "primary switches with the appearance");
    /// ```
    pub fn shadcn() -> Self {
        let mut t = Tailwind::default();
        for (name, light, dark) in COLORS {
            t.colors.insert(
                name.to_owned(),
                Paint::adaptive(parse_color(light), parse_color(dark)),
            );
        }
        // `--radius: 0.625rem`, and Tailwind's radii from it.
        for (name, px) in [("sm", 6.0), ("md", 8.0), ("lg", 10.0), ("xl", 14.0)] {
            t.radii.insert(name.to_owned(), px);
        }
        t.border_color = t.colors["border"];
        t.ring_color = t.colors["ring"].faded(0.5);
        for (name, classes) in COMPONENTS {
            t.components.insert(
                (*name).to_owned(),
                classes.split_whitespace().collect::<Vec<_>>().join(" "),
            );
        }
        t
    }
}
