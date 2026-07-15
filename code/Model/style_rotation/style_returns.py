from imports import *


class StyleReturnBuilder:
    """由t0风格暴露与t1->t2股票收益构造多空风格收益。"""

    def build(self, exposures: pd.DataFrame, labels: pd.DataFrame, styles: list[str]) -> pd.DataFrame:
        merged = exposures.merge(labels, on=["factor_date", "symbol"], how="inner", validate="one_to_one")
        rows = []
        for date, daily in merged.groupby("factor_date", sort=True):
            for style in styles:
                group = daily[[style, "label", "label_entry_date", "label_exit_date"]].dropna()
                if len(group) < 10:
                    continue
                rank = group[style].rank(pct=True, method="average")
                spread = group.loc[rank >= .8, "label"].mean() - group.loc[rank <= .2, "label"].mean()
                rows.append({"factor_date": date, "style": style, "style_return": spread,
                             "style_entry_date": group.label_entry_date.max(),
                             "style_exit_date": group.label_exit_date.max()})
        return pd.DataFrame(rows).sort_values(["factor_date", "style"]).reset_index(drop=True)
