import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def make_top_perc_chart(df, target_label, plot_first_percentiles=False, plot_perc_overall=False):

    df = df.sort_values(by='unit_risk_score', ascending=False)

    df['cumulative_true'] = df[target_label].cumsum()
    df['cumulative_count'] = range(1, len(df) + 1)

    # 4. Calculate percentages
    df['percentage_of_true'] = df['cumulative_true'] / df['cumulative_count'] * 100
    percentage_of_true_overall = df['survey_label'].sum() / df['survey_label'].count() * 100
    df['percentage_of_records'] = df['cumulative_count'] / len(df) * 100
    fig, ax1 = plt.subplots()

    # 5. Plotting
    ax1.plot(df['percentage_of_records'], df['percentage_of_true'])
    ax1.axhline(y=percentage_of_true_overall, color='orange', linestyle='--')

    if plot_first_percentiles:
        mask_5 = df['percentage_of_records'] <= 5
        percentage_of_true_5 = df[mask_5]['survey_label'].sum() / df[mask_5]['survey_label'].count() * 100
        mask_10 = df['percentage_of_records'] <= 10
        percentage_of_true_10 = df[mask_10]['survey_label'].sum() / df[mask_10]['survey_label'].count() * 100
        mask_15 = df['percentage_of_records'] <= 15
        percentage_of_true_15 = df[mask_15]['survey_label'].sum() / df[mask_15]['survey_label'].count() * 100
        mask_20 = df['percentage_of_records'] <= 20
        percentage_of_true_20 = df[mask_20]['survey_label'].sum() / df[mask_20]['survey_label'].count() * 100
        mask_25 = df['percentage_of_records'] <= 25
        percentage_of_true_25 = df[mask_25]['survey_label'].sum() / df[mask_25]['survey_label'].count() * 100

        ax1.axvline(x=5, color='c', linestyle='--', alpha=0.3)
        ax1.axvline(x=10, color='c', linestyle='--', alpha=0.3)
        ax1.axvline(x=15, color='c', linestyle='--', alpha=0.3)
        ax1.axvline(x=20, color='c', linestyle='--', alpha=0.3)
        #ax1.axvline(x=25, color='c', linestyle='--', alpha=0.3)

        # Add text near the vertical line
        ax1.text(50, percentage_of_true_overall - 5, 'Total Percentage of Artificial Fakes', rotation=0,
                 verticalalignment='center', color='black')

        ax1.text(2, 20, f'{str(round(percentage_of_true_5))}% within 5%', rotation=90, verticalalignment='center',
                 color='black')
        ax1.text(7, 30, f'{str(round(percentage_of_true_10))}% within 10%', rotation=90, verticalalignment='center',
                 color='black')
        ax1.text(15, 70, f'{str(round(percentage_of_true_15))}% within 15%', rotation=90, verticalalignment='center',
                 color='black')
        ax1.text(22, 75, f'{str(round(percentage_of_true_20))}% within 20%', rotation=90, verticalalignment='center',
                 color='black')

    if plot_perc_overall:
        ax2 = ax1.twinx()
        ax2.plot(df['percentage_of_records'], df['cumulative_true'] / df[target_label].sum() * 100, color='green', alpha=0.3)
        ax2.set_ylim([0, 100])
        ax2.set_ylabel('Percentage overall of artificial fakes (%)')

    ax1.set_ylim([0, 100])
    ax1.set_xlabel('Top N% of Interviews (%)')
    ax1.set_ylabel('Percentage of artificial fakes (%)')
    plt.title('Percentage of Artificial Fakes in Top Records')
    ax1.grid(True)
    plt.show()





def make_score_perc_chart(df, target_label, plot_first_percentiles=False, plot_perc_overall=False):
    df = df.sort_values(by='unit_risk_score', ascending=False)


    df['cumulative_count'] = range(1, len(df) + 1)

    df['percentage_of_records'] = df['cumulative_count'] / len(df) * 100
    fig, ax1 = plt.subplots()

    # 5. Plotting
    ax1.plot(df['percentage_of_records'], df['percentage_of_true'])




    if plot_perc_overall:
        ax2 = ax1.twinx()
        ax2.plot(df['percentage_of_records'], df['cumulative_true'] / df[target_label].sum() * 100, color='green', alpha=0.3)
        ax2.set_ylim([0, 100])
        ax2.set_ylabel('Percentage overall of artificial fakes (%)')

    ax1.set_ylim([0, 100])
    ax1.set_xlabel('Top N% of Interviews (%)')
    ax1.set_ylabel('Percentage of artificial fakes (%)')
    plt.title('Percentage of Artificial Fakes in Top Records')
    ax1.grid(True)
    plt.show()
