"""
Generate outcome distribution figures for MPECLib, MacMPEC, and NOSBENCH benchmarks.
Produces EPS files following OMS/T&F journal figure guidelines:
- Vertical bar charts with clear labels
- Black and white compatible (with patterns/hatching)
- Professional academic style
- Consistent formatting across all benchmark figures
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Agg')  # Non-interactive backend

# Set font to match LaTeX/academic style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'text.usetex': False,  # Set True if LaTeX available
})


def create_outcome_bar_chart(data, total_problems, benchmark_name, filename):
    """
    Create a vertical bar chart for benchmark outcome distribution.
    Style follows OMS/T&F academic journal guidelines.
    """

    # Categories and values
    categories = list(data.keys())
    values = list(data.values())
    percentages = [v / total_problems * 100 for v in values]

    # Create figure with appropriate size for single-column journal
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    # Color scheme: grayscale-compatible with distinct patterns
    colors = []
    hatches = []
    for cat in categories:
        if 'B-stationary' in cat:
            colors.append('#2E7D32')  # Dark green
            hatches.append('')
        elif 'C-stationary' in cat:
            colors.append('#1976D2')  # Blue
            hatches.append('//')
        elif 'Timeout' in cat:
            colors.append('#FF8F00')  # Orange
            hatches.append('\\\\')
        elif 'infeasible' in cat.lower():
            colors.append('#C62828')  # Red
            hatches.append('xx')
        elif 'failure' in cat.lower() or 'NLP' in cat:
            colors.append('#7B1FA2')  # Purple
            hatches.append('..')
        else:
            colors.append('#616161')  # Grey
            hatches.append('--')

    # Create vertical bar chart
    x_pos = np.arange(len(categories))
    bar_width = 0.6
    bars = ax.bar(x_pos, values, width=bar_width, color=colors, edgecolor='black', linewidth=0.8)

    # Add hatching for grayscale compatibility
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    # Add value labels on top of bars
    for i, (bar, val, pct) in enumerate(zip(bars, values, percentages)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + max(values) * 0.02,
               f'{val}\n({pct:.1f}%)', ha='center', va='bottom',
               fontsize=8, fontweight='normal', color='black')

    # Customize axes
    ax.set_xticks(x_pos)
    # Wrap long labels
    wrapped_labels = []
    for cat in categories:
        if len(cat) > 12:
            # Add line break for long labels
            words = cat.replace('-', '- ').split()
            if len(words) > 1:
                mid = len(words) // 2
                wrapped_labels.append('\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])]))
            else:
                wrapped_labels.append(cat)
        else:
            wrapped_labels.append(cat)
    ax.set_xticklabels(wrapped_labels, rotation=0, ha='center')
    ax.set_ylabel('Number of problems')
    ax.set_ylim(0, max(values) * 1.20)

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add subtle grid on y-axis only
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    # Tight layout
    plt.tight_layout()

    # Save as EPS (required by T&F)
    plt.savefig(filename, format='eps', bbox_inches='tight', dpi=300)
    print(f"Saved: {filename}")

    # Also save as PDF for convenience
    pdf_filename = filename.replace('.eps', '.pdf')
    plt.savefig(pdf_filename, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved: {pdf_filename}")

    plt.close()


# =============================================================================
# MPECLib data (92 problems)
# =============================================================================
mpeclib_data = {
    'B-stationary': 72,
    'C-stationary': 13,
    'Timeout': 7
}

# =============================================================================
# MacMPEC data (191 problems)
# =============================================================================
macmpec_data = {
    'B-stationary': 120,
    'C-stationary': 64,
    'Timeout': 6,
    'Comp.\ninfeasible': 1
}

# =============================================================================
# NOSBENCH data (603 problems)
# =============================================================================
nosbench_data = {
    'B-stationary': 471,
    'C-stationary': 84,
    'Timeout': 31,
    'NLP failure': 14,
    'Comp.\ninfeasible': 3
}


if __name__ == '__main__':
    print("="*60)
    print("Generating benchmark outcome figures (OMS/T&F style)")
    print("Vertical bar charts")
    print("="*60)

    print("\n1. Generating MPECLib outcome figure...")
    create_outcome_bar_chart(
        mpeclib_data,
        total_problems=92,
        benchmark_name='MPECLib',
        filename='mpeclib_outcomes.eps'
    )

    print("\n2. Generating MacMPEC outcome figure...")
    create_outcome_bar_chart(
        macmpec_data,
        total_problems=191,
        benchmark_name='MacMPEC',
        filename='macmpec_outcomes.eps'
    )

    print("\n3. Generating NOSBENCH outcome figure...")
    create_outcome_bar_chart(
        nosbench_data,
        total_problems=603,
        benchmark_name='NOSBENCH',
        filename='nosbench_outcomes.eps'
    )

    print("\n" + "="*60)
    print("All figures generated successfully!")
    print("="*60)
