from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PAPER_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = PAPER_DIR / "tables"


TABLE_SPECS = {
    "test_metrics": [
        "model",
        "n_obs",
        "MAE",
        "RMSE",
        "SMAPE",
        "best_order",
        "guerrero_smoothness",
        "best_lambda",
        "window",
        "alpha",
        "degree",
    ],
    "selection_metrics": [
        "model",
        "n_obs",
        "MAE",
        "RMSE",
        "SMAPE",
        "best_order",
        "guerrero_smoothness",
        "best_lambda",
        "window",
        "alpha",
        "degree",
    ],
    "model_parameters": [
        "model",
        "kind",
        "best_order",
        "guerrero_smoothness",
        "best_lambda",
        "degree",
        "window",
        "alpha",
    ],
    "roughness_metrics": ["model", "roughness_order", "realized_roughness", "test_RMSE"],
}


def _format_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    keep = [column for column in columns if column in df.columns]
    out = df[keep].copy()
    for column in out.select_dtypes(include="number").columns:
        out[column] = out[column].round(6)
    return out


def _escape_latex(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _to_latex_tabular(df: pd.DataFrame) -> str:
    alignment = "l" * len(df.columns)
    lines = [r"\begin{center}", r"\resizebox{\linewidth}{!}{%", rf"\begin{{tabular}}{{{alignment}}}", r"\toprule"]
    lines.append(" & ".join(_escape_latex(column) for column in df.columns) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        lines.append(" & ".join(_escape_latex(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{center}"])
    return "\n".join(lines) + "\n"


def make_latex_tables(tables_dir: Path = TABLES_DIR) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    for stem, columns in TABLE_SPECS.items():
        csv_path = tables_dir / f"{stem}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV table: {csv_path}")
        df = pd.read_csv(csv_path)
        formatted = _format_frame(df, columns)
        latex = _to_latex_tabular(formatted)
        (tables_dir / f"{stem}.tex").write_text(latex)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from CSV tables.")
    parser.parse_args()
    make_latex_tables()
    print("Saved LaTeX tables.")


if __name__ == "__main__":
    main()
