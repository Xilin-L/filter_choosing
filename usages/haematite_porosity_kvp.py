import multiprocessing as mp
import itertools
import pandas as pd
import numpy as np
import materialPropertiesData as mpd
from filterPerformance import getRuleOfThumbFilterThickness, setSpectrum, xs
from functools import partial

def process_combination(sample_mat, args):
    sampleThicknessMm, porosity, kvp = args
    contrast_agents = ["air", "spt0.27mol", "spt0.63mol"]

    # 1. calculate filter thickness for air only
    sample_material_air = [[sample_mat, "air"], [1 - porosity, porosity]]
    ft_air = getRuleOfThumbFilterThickness(
        kVpeak=kvp,
        filterMaterial="Fe",
        sampleMaterial=sample_material_air,
        sampleDiameterMm=sampleThicknessMm
    )

    transmissions = {}
    for agent in contrast_agents:
        # reuse ft_air for every agent
        energyKeV, spectrum = setSpectrum(kvp, "Fe", ft_air)

        # sample transmission
        w, s, d = mpd.getMaterialProperties([[sample_mat, agent], [1 - porosity, porosity]])
        sample_trans = xs.calcTransmission(energyKeV, w, s, d, sampleThicknessMm / 10)

        # agent transmission
        aw, asy, ad = mpd.getMaterialProperties(agent)
        if sampleThicknessMm >= 15:
            agent_trans = xs.calcTransmission(energyKeV, aw, asy, ad, 0.2)
        else:
            agent_trans = xs.calcTransmission(energyKeV, aw, asy, ad, 0.1)

        wall_trans = xs.calcTransmission(energyKeV, [1.0], ["Al"], 2.7, 0.2)

        transmissions[agent] = np.sum(spectrum * sample_trans * agent_trans * wall_trans) / np.sum(spectrum)

    return [
        sampleThicknessMm,
        porosity,
        kvp,
        ft_air,
        transmissions["air"],
        transmissions["spt0.27mol"],
        transmissions["spt0.63mol"]
    ]


if __name__ == "__main__":
    thickness_values = [12, 15, 17, 20, 25]
    porosity_values = [0.0, 0.1, 0.2, 0.4, 0.6]
    kvp_values = list(range(300, 140, -10))
    combos = list(itertools.product(thickness_values, porosity_values, kvp_values))

    num_cores = max(1, mp.cpu_count() // 2)
    writer = pd.ExcelWriter("haematite_goethite.xlsx", engine='xlsxwriter')
    with mp.Pool(num_cores) as pool:
        for sample in ["haematite", "goethite"]:
            func = partial(process_combination, sample)
            results = pool.map(func, combos)
            df = pd.DataFrame(results, columns=[
                'SampleThicknessMm', 'Porosity', 'kVp',
                'FilterThicknessAir',
                'TransmissionAir', 'TransmissionSPT0.27', 'TransmissionSPT0.63'
            ])

            for ag in ['SPT0.27', 'SPT0.63']:
                col = f'Transmission{ag}'
                diff_col = f'Diff{ag}'
                # compute the difference from TransmissionAir
                df[diff_col] = df[col] - df['TransmissionAir']
                # move the diff column right after its source column
                loc = df.columns.get_loc(col) + 1
                df.insert(loc, diff_col, df.pop(diff_col))

            # round filter thickness to 1 decimal
            filter_cols = [c for c in df.columns if c.startswith('FilterThickness')]
            df[filter_cols] = df[filter_cols].round(1)

            # round transmissions and differences to 3 decimals
            trans_cols = [c for c in df.columns if c.startswith('Transmission') or c.startswith('Diff')]
            df[trans_cols] = df[trans_cols].round(3)

            base = ['SampleThicknessMm', 'Porosity', 'kVp']
            metrics_map = {
                'Air': ['FilterThickness', 'Transmission'],
                'SPT0.27': ['Transmission', 'Diff'],
                'SPT0.63': ['Transmission', 'Diff']
            }

            tuples = [('', col) for col in base]
            for agent, metrics in metrics_map.items():
                for metric in metrics:
                    tuples.append((agent, metric))

            df.columns = pd.MultiIndex.from_tuples(tuples, names=['Agent', 'Metric'])

            sheet = sample.capitalize()
            ws = writer.book.add_worksheet(sheet)
            writer.sheets[sheet] = ws

            big = writer.book.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'bg_color': '#D7E4BC'})
            sub = writer.book.add_format({'bold': True, 'align': 'center', 'bg_color': '#DDEBF7'})

            # count how many metrics each agent has
            agent_counts = df.columns.get_level_values('Agent').value_counts()

            # first‐level spans
            spans = [('', len(base))] + [
                (agent, agent_counts.get(agent, 0))
                for agent in ['Air', 'SPT0.27', 'SPT0.63']
            ]

            # merge and write first header row
            col = 0
            for title, span in spans:
                ws.merge_range(0, col, 0, col + span - 1, title, big)
                col += span

            # write second header row by walking the MultiIndex
            col = 0
            for agent, metric in df.columns:
                ws.write(1, col, metric, sub)
                col += 1

            # write data rows and adjust widths as before…
            for r, row in enumerate(df.values, start=2):
                ws.write_row(r, 0, row)

            for idx in range(df.shape[1]):
                w = max(
                    df.iloc[:, idx].astype(str).map(len).max(),
                    len(df.columns[idx][1])
                ) + 2
                ws.set_column(idx, idx, w)

            ws.freeze_panes(2, 0)

    writer.close()
    print("Results saved to `haematite_goethite.xlsx`")
