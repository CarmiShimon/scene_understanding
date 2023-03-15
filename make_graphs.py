import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from numpy import datetime64
plt.interactive(False)
matplotlib.use('TkAgg')

if __name__ == '__main__':
    upload_dates = r'../videos/orig_videos'
    analysis_dir = r'../videos/summary'
    sub_dirs = os.listdir(analysis_dir)

    for sub_dir in sub_dirs:
        if sub_dir != 'Child 102':
            continue
        upload_dates_file = os.path.join(upload_dates, sub_dir) + '/upload_dates.xlsx'
        upload_dates_file = upload_dates_file.replace('\\', '/')
        upload_dates_df = pd.read_excel(upload_dates_file)
        upload_dates_df['dates'] = pd.to_datetime(upload_dates_df["date"], dayfirst=True)
        upload_dates_df.sort_values(by='dates', inplace=True)
        out_dir = os.path.join(analysis_dir, sub_dir)
        out_dir = out_dir.replace('\\', '/')

        times = []
        communicative_speech = []
        communicative_times = []
        reinforcement = []
        no_reinforcement = []
        double_demand = []
        child_adult_seconds = []
        for i, row in upload_dates_df.iterrows():
            if isinstance(row['number'], float):
                filename = int(row['number'])
            else:
                filename = row['number']
            analysis_file = f"{out_dir}/{filename}.csv"
            if os.path.isfile(analysis_file):
                analysis_df = pd.read_csv(analysis_file)
                # filter up to april month
                if row['dates'].year >= 2022 and row['dates'].month >= 4:
                    continue
                row['dates'] = row['dates'].replace(year=row['dates'].year, month=row['dates'].day, day=row['dates'].month)
                if len(times) > 1 and datetime64(row['dates']) in times:
                    idx = np.where(times == datetime64(row['dates']))[0][0]
                    communicative_speech[idx] += analysis_df.iloc[3].values[1] / (analysis_df.iloc[4].values[1]/60)
                    communicative_times[idx] += analysis_df.iloc[3].values[1]
                    # no_reinforcement[idx] += analysis_df.iloc[1].values[1] / analysis_df.iloc[3].values[
                    # reinforcement[idx] += analysis_df.iloc[0].values[1] / (analysis_df.iloc[4].values[1]/60)
                    # no_reinforcement[idx] += analysis_df.iloc[1].values[1] / (analysis_df.iloc[4].values[1]/60)
                    # double_demand[idx] += analysis_df.iloc[2].values[1] / (analysis_df.iloc[4].values[1]/60)
                    reinforcement[idx] += analysis_df.iloc[0].values[1]
                    no_reinforcement[idx] += analysis_df.iloc[1].values[1]
                    double_demand[idx] += analysis_df.iloc[2].values[1]
                else:
                    times.append(datetime64(row['dates']))
                    communicative_speech.append(analysis_df.iloc[3].values[1] / (analysis_df.iloc[4].values[1]/60))
                    communicative_times.append(analysis_df.iloc[3].values[1])
                    # reinforcement.append(analysis_df.iloc[0].values[1] / (analysis_df.iloc[4].values[1]/60))
                    # no_reinforcement.append(analysis_df.iloc[1].values[1] / (analysis_df.iloc[4].values[1]/60))
                    # double_demand.append(analysis_df.iloc[2].values[1] / (analysis_df.iloc[4].values[1]/60))

                    reinforcement.append(analysis_df.iloc[0].values[1])
                    no_reinforcement.append(analysis_df.iloc[1].values[1])
                    double_demand.append(analysis_df.iloc[2].values[1])

        reinforcement = np.asarray(reinforcement) / np.asarray(communicative_times)
        no_reinforcement = np.asarray(no_reinforcement) / np.asarray(communicative_times)
        double_demand = np.asarray(double_demand) / np.asarray(communicative_times)

        fig = plt.figure(figsize = (10, 5))
        plt.grid()
        plt.plot_date(times, communicative_speech, markersize=7)
        # plt.bar(times, communicative_speech, color='maroon',
        #         width=0.8)
        plt.xlabel("Dates")
        plt.ylabel("Communicative Speech per analyzable minutes")
        plt.title(f"{sub_dir} Communicative Speech progress")
        fig.autofmt_xdate()
        plt.savefig(f'{out_dir}/Communicative.png')

        fig = plt.figure(figsize = (10, 5))
        plt.grid()
        plt.plot_date(times, reinforcement*100, markersize=7)
        # plt.bar(times, reinforcement, color='maroon',
        #         width=0.8)
        plt.xlabel("Dates")
        plt.ylabel("Reinforcement Ratio [%]")
        plt.ylim([-5, 105])
        plt.title(f"{sub_dir}: Adult Reinforcement")
        fig.autofmt_xdate()
        plt.savefig(f'{out_dir}/reinforcement.png')

        fig = plt.figure(figsize = (10, 5))
        plt.grid()
        plt.plot_date(times, no_reinforcement*100, markersize=7)
        # plt.bar(times, no_reinforcement, color='maroon',
        #         width=0.8)
        plt.xlabel("Dates")
        plt.ylabel("No - Reinforcement Ratio [%]")
        plt.ylim([-5, 105])
        plt.title(f"{sub_dir}: Adult No-Reinforcement")
        fig.autofmt_xdate()
        plt.savefig(f'{out_dir}/No_Reinforcement.png')

        fig = plt.figure(figsize = (10, 5))
        plt.grid()
        plt.plot_date(times, double_demand*100, markersize=7)
        # plt.bar(times, double_demand, color='maroon',
        #         width=0.8)
        plt.xlabel("Dates")
        plt.ylabel("Double Demand Ratio [%]")
        plt.ylim([-5, 105])
        plt.title(f"{sub_dir}: Adult Double Demand")
        fig.autofmt_xdate()
        plt.savefig(f'{out_dir}/Double_Demand.png')
        fig.clf()
        plt.close()
    print('Done creating graphs')



