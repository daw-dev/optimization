use std::ops::Range;

#[derive(Clone)]
pub struct Iterations(pub usize);

#[derive(Clone)]
pub struct Precision(pub f64);

#[derive(Clone)]
pub struct Average;

#[derive(Debug, Clone)]
pub struct UniformSample {
    range: Range<f64>,
    points: usize,
    current_point: usize,
}

impl UniformSample {
    pub fn new(range: Range<f64>, points: usize) -> Self {
        Self {
            range,
            points,
            current_point: 0,
        }
    }
}

impl Iterator for UniformSample {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        (self.current_point <= self.points).then(|| {
            let res = self.range.start
                + (self.range.end - self.range.start) * self.current_point as f64
                    / self.points as f64;
            self.current_point += 1;
            res
        })
    }
}

// impl<X, Y> Optimizer<X, Y, Range<f64>, f64> for Average {
//     fn optimize<F: Function<X, Y>>(
//         self,
//         _func: &F,
//         starting_guess: Range<f64>,
//     ) -> impl crate::optimizer::OptimizationResult<Guess = f64> {
//         Optimization::new(std::iter::once((starting_guess.start + starting_guess.end) / 2.0))
//     }
// }

pub fn save_dashboard<P: AsRef<std::path::Path>>(
    path: P,
    title: &str,
    plots: &[(&str, &plotly::Plot)],
) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(path)?;

    writeln!(
        file,
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{0}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Outfit', sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
        }}
        .plot-container {{
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
            transition: all 0.3s ease;
        }}
        .plot-container:hover {{
            border-color: rgba(99, 102, 241, 0.4);
            box-shadow: 0 15px 35px -5px rgba(99, 102, 241, 0.15);
        }}
    </style>
</head>
<body class="min-h-screen p-6 md:p-12">
    <div class="max-w-7xl mx-auto">
        <header class="mb-12 text-center md:text-left border-b border-slate-800 pb-8">
            <h1 class="text-4xl md:text-5xl font-extrabold bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent mb-2">{0}</h1>
            <p class="text-slate-400 text-lg">Interactive Visualization Dashboard</p>
        </header>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">"#,
        title
    )?;

    for (i, (plot_title, plot)) in plots.iter().enumerate() {
        let plot_json = plot.to_json();
        writeln!(
            file,
            r#"
            <div class="plot-container p-6 flex flex-col">
                <h2 class="text-xl font-bold mb-4 text-indigo-300 flex items-center gap-2">
                    <span class="inline-block w-2.5 h-2.5 rounded-full bg-indigo-500"></span>
                    {0}
                </h2>
                <div id="plot-{1}" class="w-full h-[600px]"></div>
                <script>
                    (function() {{
                        let spec = {2};
                        spec.layout = spec.layout || {{}};
                        spec.layout.paper_bgcolor = 'rgba(0,0,0,0)';
                        spec.layout.plot_bgcolor = 'rgba(0,0,0,0)';
                        spec.layout.font = spec.layout.font || {{}};
                        spec.layout.font.color = '#cbd5e1';
                        spec.layout.font.family = 'Outfit, sans-serif';
                        
                        // Fix legend overlap by placing it horizontally at the top
                        spec.layout.legend = spec.layout.legend || {{}};
                        spec.layout.legend.orientation = 'h';
                        spec.layout.legend.yanchor = 'bottom';
                        spec.layout.legend.y = 1.02;
                        spec.layout.legend.xanchor = 'left';
                        spec.layout.legend.x = 0;
                        
                        // Increase viewport size by reducing margins
                        if (spec.layout.scene) {{
                            spec.layout.margin = spec.layout.margin || {{ l: 0, r: 0, t: 0, b: 0 }};
                            spec.layout.scene.xaxis = spec.layout.scene.xaxis || {{}};
                            spec.layout.scene.xaxis.gridcolor = '#334155';
                            spec.layout.scene.yaxis = spec.layout.scene.yaxis || {{}};
                            spec.layout.scene.yaxis.gridcolor = '#334155';
                            spec.layout.scene.zaxis = spec.layout.scene.zaxis || {{}};
                            spec.layout.scene.zaxis.gridcolor = '#334155';
                        }} else {{
                            spec.layout.margin = spec.layout.margin || {{ l: 50, r: 30, t: 40, b: 50 }};
                        }}
                        Plotly.newPlot('plot-{1}', spec.data, spec.layout, {{responsive: true}});
                    }})();
                </script>
            </div>"#,
            plot_title, i, plot_json
        )?;
    }

    writeln!(
        file,
        r#"
        </div>
    </div>
</body>
</html>"#
    )?;

    Ok(())
}

pub fn prompt_and_open_dashboard<P: AsRef<std::path::Path>>(path: P) {
    use std::io::{self, Write};
    use std::process::Command;

    print!("\nDo you want to open the plot? [y/N]: ");
    let _ = io::stdout().flush();

    let mut input = String::new();
    if io::stdin().read_line(&mut input).is_ok() {
        let trimmed = input.trim().to_lowercase();
        if trimmed == "y" || trimmed == "yes" {
            let path_str = path.as_ref().to_string_lossy().to_string();
            println!("Opening {}...", path_str);
            let status = Command::new("firefox")
                .arg(&path_str)
                .spawn()
                .or_else(|_| Command::new("xdg-open").arg(&path_str).spawn());
            
            if let Err(e) = status {
                println!("Failed to open browser automatically: {}", e);
            }
        }
    }
}
