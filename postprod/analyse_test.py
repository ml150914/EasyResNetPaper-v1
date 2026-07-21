import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def RoC(df1, df2, column, var, label, save = True, thr=0.0001, color = 'b'):
    """                                                                                                                                                                                      
    Calculates the cumulative distribution for the specified column in a pandas DataFrame.                                                                                                   
    The cumulative distribution indicates how many elements are above a given threshold.                                                                                                     
                                                                                                                                                                                             
    Parameters:                                                                                                                                                                              
    df (pd.DataFrame): The input DataFrame.                                                                                                                                                  
    column (str): The name of the column to calculate the cumulative distribution for. Defaults to 'b'.                                                                                      
                                                                                                                                                                                             
    Returns:                                                                                                                                                                                 
    pd.DataFrame: A DataFrame with two columns: 'threshold' and 'count_above_threshold'.                                                                                                     
    """
    if column not in df1.columns:
        raise ValueError(f"Column '{column}' not found in the DataFrame.")

    # Drop missing values and sort unique values of the column                                                                                                                               
    data1 = df1[column].dropna()
    data2 = df2[column].dropna()
    #thresholds = np.sort(data.unique())                                                                                                                                                     
    thresholds = np.arange(min(data1),max(data1), thr)

    # Calculate the cumulative distribution                                                                                                                                                  
    count_above_threshold1 = [np.sum(data1 > threshold) / len(data1) for threshold in thresholds]
    count_above_threshold2 =  [np.sum(data2 > threshold) / len(data2) for threshold in thresholds]

    # Create a DataFrame for the cumulative distribution                                                                                                                                     
    cumulative_df = pd.DataFrame({
        'threshold': thresholds,
        'count_above_threshold1': count_above_threshold1,
        'count_above_threshold2': count_above_threshold2
    })

    if save:
        cumulative_df.to_csv(path_data + f'/{var}_RoC_{label}.csv', index=False)

    # Plot the cumulative distribution                                                                                                                                                       
    plt.plot(cumulative_df['count_above_threshold1'], cumulative_df['count_above_threshold2'], label=label, c = color)

    return cumulative_df

file = '/home/lorenzo-mobilia/public_html/EasyResNetPaper-v1/S2/result_correct_seed_70k/test_scores.csv'
path_data = '/home/lorenzo-mobilia/public_html/EasyResNetPaper-v1/S2/result_correct_seed_70k/'

df = pd.read_csv(file)

df_noise = df[df['Label'] == 'Noise']
df_inj = df[df['Label'] == 'Injection']

RoC(df_noise, df_inj, 'Inj_Prob', 'Roc', 'CNN output', thr = 0.00001, color = 'steelblue')
RoC(df_noise, df_inj, 'max_snr', 'Roc', r'max $\rho$', thr = 0.005, color = 'lightgreen')

if 'max_rwsnr' in df.columns:
    RoC(df_noise, df_inj, 'max_rwsnr', 'Roc', r'max rw_$\rho$', thr = 0.0005, color = 'red')
else:
    print('max_rwsnr not present')
# Final plot settings                                                                                                                                                                        
plt.title('Cumulative Distribution (Counts Above Threshold)')
plt.xlabel('FaP')
plt.ylabel('Number of Injections')
plt.grid(True)
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.ylim([0.1, 1.1])                                                                                                                                                                       
plt.savefig(path_data + 'Roc_Superimposed.png')
plt.close()


