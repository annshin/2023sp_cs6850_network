import plotly.graph_objects as go
import pandas as pd
from copy import deepcopy


def plot_interdisciplinary_combinations_v2(data_df, year, discipline_lower):
    interdisciplinary_combinations = data_df

    # Create and sort lists of unique disciplines for citing (left) and cited (right) columns
    unique_citing_disciplines = sorted(list(interdisciplinary_combinations['paper_discipline'].unique()))
    unique_cited_disciplines = sorted(list(interdisciplinary_combinations['cited_paper_discipline'].unique()))

    # Create dictionaries to map disciplines to indices for citing (left) and cited (right) columns
    citing_discipline_index_map = {discipline: i for i, discipline in enumerate(unique_citing_disciplines)}
    cited_discipline_index_map = {discipline: len(unique_citing_disciplines) + i for i, discipline in enumerate(unique_cited_disciplines)}

    # Create lists for source, target, and value columns
    source = []
    target = []
    value = []

    for _, row in interdisciplinary_combinations.iterrows():
        source.append(citing_discipline_index_map[row['paper_discipline']])
        target.append(cited_discipline_index_map[row['cited_paper_discipline']])
        value.append(row['count'])

    # Define the color mapping for disciplines
    color_mapping = {
        'Computer Science': 'blue',
        'Statistics': 'green',
        'Physics': 'rgba(31,119,180,1)',
        'Materials Science': 'rgba(255,127,14,1)',
        'Philosphy': 'rgba(44,160,44,1)',
        'Chemistry': 'rgba(214,39,40,1)',
        'Art': 'rgba(148,103,189,1)',
        'Geology': 'rgba(140,86,75,1)',
        'Biology': 'rgba(227,119,194,1)',
        'Environmental Science': 'rgba(127,127,127,1)',
        'History': 'rgba(188,189,34,1)',
        'Geography': 'rgba(23,190,207,1)',
        'Medicine': 'rgba(166,77,121,1)',
        'Engineering': 'rgba(84,153,199,1)',
        'Mathematics': 'rgba(241,90,96,1)',
        'Psychology': 'rgba(254,219,64,1)',
        'Economics': 'rgba(244,109,67,1)',
        'Political Science': 'rgba(171,104,87,1)',
        'Sociology': 'rgba(106,183,153,1)',
        'Business': 'rgba(120,198,121,1)',
        # 'Education':
        # 'Linguistics':
        # 'Literature':
        # 'Anthropology':
        # 'Architecture':
        # 'Music':
        # 'Religion':
        # 'Law':
        # 'Agriculture':
        # 'Other':
        # Add other disciplines with corresponding colors here
    }

    # Assign colors to disciplines
    colors = []
    for discipline in unique_citing_disciplines + unique_cited_disciplines:
        colors.append(color_mapping.get(discipline, 'grey'))  # Use grey as the default color for unmapped disciplines

    # Create a Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=unique_citing_disciplines + unique_cited_disciplines,
            color=colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        )
    )])

    fig.update_layout(title_text=f'Year: {year}', font_size=10)

    # Save the figure to png
    # fig.write_image(f"interdisciplinary_combinations_ml/plots/interdisciplinary_combinations_ml_{year}.png")
    fig.write_image(f"interdisciplinary_combinations_ml_only_cs_stat/plots/interdisciplinary_combinations_ml_{year}.png")
    # fig.write_image(f"interdisciplinary_combinations_{discipline_lower}/plots/interdisciplinary_combinations_{discipline_lower}_{year}.png")


if __name__ == '__main__':


    # Plot Sankey diagrams for each year
    # year_range = ['93', '97', '98', '00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']
    # year_range = ['00']
    full_year_range = ['91', '92', '93', '94', '95', '96', '97', '98', '99', '00', 
                  '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', 
                  '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
                  '21', '22']
    # discipline = "Physics"
    # discipline = "Mathematics"
    # discipline_lower = discipline.lower()

    for year in full_year_range:
        # Read analysis/interdisciplinary/interdisciplinary_combinations_{year}.csv to df
        print(f"Year: {year}")
        # int_year = int(year)
        # data_df = pd.read_csv(f"interdisciplinary_combinations_ml/interdisciplinary_combinations_ml_{year}.csv", dtype={'year': 'str'})
        data_df = pd.read_csv(f"interdisciplinary_combinations_ml_only_cs_stat/interdisciplinary_combinations_ml_{year}.csv", dtype={'year': 'str'})
        # data_df = pd.read_csv(f"interdisciplinary_combinations_{discipline_lower}/interdisciplinary_combinations_{discipline_lower}_{year}.csv", dtype={'year': 'str'})

        # plot_interdisciplinary_combinations_v2(data_df, year, discipline_lower)
        plot_interdisciplinary_combinations_v2(data_df, year, "")