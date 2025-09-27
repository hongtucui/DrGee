from django.shortcuts import render, HttpResponse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import os
import re
from sklearn.preprocessing import StandardScaler
import concurrent.futures
from io import StringIO
from transformers import AutoTokenizer, AutoModel
import pickle
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import joblib

device = torch.device("cpu")


def home(request):
    return render(request, "ED_home.html")

def predict(request):
    return render(request, "predict.html")

def download(request):
    return render(request, "download.html")

def help(request):
    return render(request, "help.html")

def read_text_data(text_data):
    # Replace tabs and commas outside quotes with |
    text_data = re.sub(
        r'(?:"[^"]*"|\'[^\']*\')|(\s*[\t,]\s*)',
        lambda m: '|' if m.group(1) else m.group(0),
        text_data
    )
    
    # Convert text data to a file-like object
    data = StringIO(text_data)
    
    # Read data with quote handling parameters
    df = pd.read_csv(
        data, 
        sep='|',
        quotechar='"',  # Recognize content wrapped in double quotes
        skipinitialspace=True  # Ignore spaces after delimiters
    )
    
    # Clean spaces in column names and data, and remove possible remaining quotes
    df.columns = df.columns.str.strip().str.strip('"\'')
    df = df.apply(lambda col: col.map(
        lambda x: x.strip().strip('"\'') if isinstance(x, str) else x
    ))
    
    return df


def read_process_data(current_directory):
    exp = pd.read_csv(os.path.join(current_directory, 'static', 'file', 'exp.csv'))
    ess = pd.read_csv(os.path.join(current_directory, 'static', 'file', 'essential.csv'), index_col=0)
    gene_alis = pd.read_csv(os.path.join(current_directory, 'static', 'file', 'gene_alis.csv'))
    primary = pd.read_csv(os.path.join(current_directory, 'static', 'file', 'primary.csv'), index_col=0)
    secondary = pd.read_csv(os.path.join(current_directory, 'static', 'file', 'secondary_auc_ic50.csv'))
    ctd = pd.read_csv(os.path.join(current_directory, 'static', 'file', 'ctd_auc_ic50.csv'))
    gdsc1 = pd.read_csv(os.path.join(current_directory, 'static', 'file', 'gdsc1_auc_ic50.csv'))
    gdsc2 = pd.read_csv(os.path.join(current_directory, 'static', 'file', 'gdsc2_auc_ic50.csv'))
    gene_info = pd.read_csv(os.path.join(current_directory, 'static', 'file', 'gene_info.csv'))
    gene_list = [re.sub("\\..*", "", file) for file in os.listdir(os.path.join(current_directory, 'static', 'file', 'gene'))]
    actual = pd.read_csv(os.path.join(current_directory, 'static', 'file', 'actual.csv'))
    remove = pd.read_csv(os.path.join(current_directory, 'static', 'file', 'remove.csv'))
    drug_specific = pd.read_csv(os.path.join(current_directory, 'static', 'file', 'drug_specific_ratio.csv'))
    name_change = pd.read_csv(os.path.join(current_directory, 'static', 'file', 'name_change.csv'))
    brand = pd.read_csv(os.path.join(current_directory, 'static', 'file', 'brand.csv'))
    affinity = pd.read_csv(os.path.join(current_directory, 'static', 'file', 'affinity.csv'))
    return exp, ess, gene_alis, primary, secondary, ctd, gdsc1, gdsc2, gene_info, gene_list, actual, remove, drug_specific, name_change, brand, affinity

def merge_user_exp(user_dat, gene_alis, exp):
    user_dat["gene"] = user_dat["gene"].str.upper()
    user_dat = user_dat[user_dat["gene"].isin(gene_alis["alis"])]
    user_dat = pd.merge(user_dat, gene_alis, left_on="gene", right_on="alis", how='inner')
    user_dat["gene"] = user_dat["gene_symbol"]
    user_dat = user_dat.groupby('gene', as_index=False)['exp'].mean()

    merge_dat = pd.merge(exp, user_dat, on="gene", how="inner").sort_values(by="gene")
    rank_dat = merge_dat.iloc[:, 1:].rank(method='min')
    rank_dat.insert(0, 'gene', merge_dat['gene'])

    correlation = rank_dat.iloc[:, 1:-1].corrwith(rank_dat.iloc[:, -1])
    return rank_dat, correlation, user_dat


def calcu_ess_num(ess, correlation_dat, rank_dat):
    # Add correlation coefficients to the ess DataFrame
    ess['correlation'] = correlation_dat

    # Select top 44 genes by correlation
    ess_selected = ess.sort_values(by='correlation', ascending=False).iloc[:44]

    # Calculate Pearson correlation coefficient with exp column in rank_dat
    cell_line = ess_selected.index[0]
    cell_cor, cell_pvalue = pearsonr(rank_dat['exp'], rank_dat[cell_line])

    # Get all unique genes
    unique_genes = rank_dat['gene'].unique()

    # Create a DataFrame to store final weighted means
    sample_prob = pd.DataFrame(
        {'gene': [''] * len(unique_genes), 'Weighted_Mean': [0.0] * len(unique_genes)}
    )

    # Iterate through all unique genes to calculate weighted means
    for i, gene in enumerate(unique_genes):
        if gene not in ess_selected.columns:
            weighted_mean = np.nan
        else:
            gene_data = ess_selected[gene]
            if all(pd.isna(gene_data)):
                weighted_mean = np.nan
            else:
                correlation = ess_selected['correlation']
                weighted_mean = np.sum(gene_data * correlation) / np.sum(correlation)

        sample_prob.loc[i, 'gene'] = gene
        sample_prob.loc[i, 'Weighted_Mean'] = weighted_mean

    # Reset index
    sample_prob.reset_index(drop=True, inplace=True)

    return cell_line, cell_cor, cell_pvalue, sample_prob


def prepare_drug_info(*args):
    primary, secondary, ctd, gdsc1, gdsc2, cell_line = args
    primary_database = []
    primary_drug = []
    primary_effect = pd.DataFrame()
    secondary_database = []
    secondary_drug = []
    secondary_effect = pd.DataFrame()
    ctd_database = []
    ctd_drug = []
    ctd_effect = pd.DataFrame()
    gdsc1_database = []
    gdsc1_drug = []
    gdsc1_effect = pd.DataFrame()
    gdsc2_database = []
    gdsc2_drug = []
    gdsc2_effect = pd.DataFrame()

    if cell_line in primary.index:
        primary_database = list(primary.columns)
        primary_drug = list(primary.columns[(primary.loc[cell_line] <= -1.74) & (~primary.loc[cell_line].isna())])
        primary_effect = primary.loc[cell_line].transpose().reset_index()
        primary_effect.rename(columns={'index': 'drug', cell_line: 'drug_effect'}, inplace=True)
        primary_effect['source'] = 'primary'

    if cell_line in secondary['cell_line'].values:
        secondary_subset = secondary[secondary['cell_line'] == cell_line]
        secondary_database = list(secondary_subset['drug'])
        secondary_drug = list(
            secondary_subset[(secondary_subset['ic50'] <= 0.5) & (~secondary_subset['ic50'].isna()) | (
                    secondary_subset['auc'] < 0.5) & (~secondary_subset['auc'].isna())]['drug'])
        secondary_effect = secondary_subset[['drug', 'auc']].rename(columns={'auc': 'drug_effect'})
        secondary_effect['source'] = 'secondary'

    if cell_line in ctd['cell_line'].values:
        ctd_subset = ctd[ctd['cell_line'] == cell_line]
        ctd_database = list(ctd_subset['drug'])
        ctd_drug = list(ctd_subset[
                            (ctd_subset['ic50'] <= 0.5) & (~ctd_subset['ic50'].isna()) | (ctd_subset['auc'] < 0.5) & (
                                ~ctd_subset['auc'].isna())]['drug'])
        ctd_effect = ctd_subset[['drug', 'auc']].rename(columns={'auc': 'drug_effect'})
        ctd_effect['source'] = 'ctd'

    if cell_line in gdsc1['cell_line'].values:
        gdsc1_subset = gdsc1[gdsc1['cell_line'] == cell_line]
        gdsc1_database = list(gdsc1_subset['drug'])
        gdsc1_drug = list(gdsc1_subset[(gdsc1_subset['ic50'] <= 0.5) & (~gdsc1_subset['ic50'].isna()) | (
                gdsc1_subset['auc'] < 0.5) & (~gdsc1_subset['auc'].isna())]['drug'])
        gdsc1_effect = gdsc1_subset[['drug', 'auc']].rename(columns={'auc': 'drug_effect'})
        gdsc1_effect['source'] = 'gdsc1'

    if cell_line in gdsc2['cell_line'].values:
        gdsc2_subset = gdsc2[gdsc2['cell_line'] == cell_line]
        gdsc2_database = list(gdsc2_subset['drug'])
        gdsc2_drug = list(gdsc2_subset[(gdsc2_subset['ic50'] <= 0.5) & (~gdsc2_subset['ic50'].isna()) | (
                gdsc2_subset['auc'] < 0.5) & (~gdsc2_subset['auc'].isna())]['drug'])
        gdsc2_effect = gdsc2_subset[['drug', 'auc']].rename(columns={'auc': 'drug_effect'})
        gdsc2_effect['source'] = 'gdsc2'

        # Collect all qualifying drugs
    all_drugs = set(primary_drug + secondary_drug + ctd_drug + gdsc1_drug + gdsc2_drug)
    recommended_drugs = []

    for drug in all_drugs:
        # Check if drug meets criteria in all databases it appears in
        if (drug in primary_drug or drug not in primary_database) and \
                (drug in secondary_drug or drug not in secondary_database) and \
                (drug in ctd_drug or drug not in ctd_database) and \
                (drug in gdsc1_drug or drug not in gdsc1_database) and \
                (drug in gdsc2_drug or drug not in gdsc2_database):
            recommended_drugs.append(drug)
    
    for drug in all_drugs:
        if drug not in recommended_drugs:
            add_drug = True

            # Check primary database condition
            if drug in primary_database:
                drug_effect_value = primary.loc[cell_line, drug]
                if drug_effect_value > -1:
                    add_drug = False

            # Check each dataset for ic50 and auc conditions
            for dataset, name in zip([secondary, ctd, gdsc1, gdsc2],
                                     ['secondary', 'ctd', 'gdsc1', 'gdsc2']):
                if drug in locals().get("{}_database".format(name), {}):
                    ic50_value = dataset.loc[(dataset['cell_line'] == cell_line) & (dataset['drug'] == drug), 'ic50']
                    # auc_value = dataset.loc[(dataset['cell_line'] == cell_line) & (dataset['drug'] == drug), 'auc']

                    if ic50_value.isna().iloc[0] or ic50_value.iloc[0] > 1:
                        add_drug = False
                        break

                    # if auc_value.isna().iloc[0] or auc_value.iloc[0] >= 0.5:
                    #     add_drug = False
                    #     break

            # Add drug to recommended_drugs if conditions are met
            if add_drug:
                recommended_drugs.append(drug)

    # Construct final drug effect DataFrame
    drug_effect = pd.concat([primary_effect, secondary_effect, ctd_effect, gdsc1_effect, gdsc2_effect],
                            ignore_index=True)
    if not drug_effect.empty:
        drug_effect.columns = ['drug', 'drug_effect', 'source']
    else:
        drug_effect = pd.DataFrame(columns=['drug', 'drug_effect', 'source'])
        
    # Filter and output final drugs
    final_primary_drug = [drug for drug in primary_drug if drug in recommended_drugs]
    final_secondary_drug = [drug for drug in secondary_drug if drug in recommended_drugs]
    final_ctd_drug = [drug for drug in ctd_drug if drug in recommended_drugs]
    final_gdsc1_drug = [drug for drug in gdsc1_drug if drug in recommended_drugs]
    final_gdsc2_drug = [drug for drug in gdsc2_drug if drug in recommended_drugs]
        
    return final_primary_drug, final_secondary_drug, final_ctd_drug, final_gdsc1_drug, final_gdsc2_drug, drug_effect

def increase_mark(ess, cell_line, rank_dat, gene_list):
    cell_essential_gene = ess.columns[(ess.loc[cell_line] > 0.5) & (~ess.loc[cell_line].isna())].tolist()
    no_cell_essential_gene = ess.columns[(ess.loc[cell_line] <= 0.5) & (~ess.loc[cell_line].isna())].tolist()

   # gene_increase = rank_dat[(rank_dat['exp'] >= rank_dat[cell_line])].copy()
    gene_increase = rank_dat.copy()
    gene_increase.loc[:, 'rank'] = gene_increase['exp'] - gene_increase[cell_line]
    gene_increase = gene_increase[['gene', 'rank']]
    gene_increase = gene_increase[gene_increase['gene'].isin(gene_list)]

    gene_increase['category'] = "unknown"
    gene_increase.loc[gene_increase['gene'].isin(cell_essential_gene), 'category'] = "essential_gene"
    gene_increase.loc[gene_increase['gene'].isin(no_cell_essential_gene), 'category'] = "no_essential_gene"

    return gene_increase

def drug_candidate(gene_increase, current_directory, primary_drug, secondary_drug, ctd_drug, gdsc1_drug, gdsc2_drug):
    # Initialize result DataFrame with specified columns to avoid type warnings
    result_drug_gene = pd.DataFrame(columns=['gene', 'drug', 'correlation', 'p_value', 'cal_num', 'source'])

    # Collect all valid DataFrames in a list for one-time concatenation to improve performance
    valid_dfs = []

    for gene in gene_increase['gene']:
        gene_file_path = os.path.join(current_directory, 'static', 'file', 'gene', f'{gene}.csv')

        # Check if file exists to avoid read errors
        if not os.path.exists(gene_file_path):
            continue

        gene_file = pd.read_csv(gene_file_path)

        # Skip empty files to avoid concatenation warnings
        if gene_file.empty:
            continue

        # Correct logical operator precedence with parentheses to ensure proper conditions
        filtered = gene_file[
            (
                ((gene_file['source'] == "primary") & gene_file['drug'].isin(primary_drug)) |
                ((gene_file['source'] == "secondary") & gene_file['drug'].isin(secondary_drug)) |
                ((gene_file['source'] == "ctd") & gene_file['drug'].isin(ctd_drug)) |
                ((gene_file['source'] == "gdsc1") & gene_file['drug'].isin(gdsc1_drug)) |
                ((gene_file['source'] == "gdsc2") & gene_file['drug'].isin(gdsc2_drug))
            ) &
            (gene_file['correlation'] > 0.4) &
            (gene_file['p_value'] < 0.05)
        ]

        # Only add non-empty results
        if not filtered.empty:
            valid_dfs.append(filtered)

    # Concatenate all valid DataFrames at once
    if valid_dfs:
        result_drug_gene = pd.concat(valid_dfs, ignore_index=True)

    # Remove rows containing NaN
    result_drug_gene = result_drug_gene.dropna()

    return result_drug_gene
    

# def drug_candidate(gene_increase, current_directory, primary_drug, secondary_drug, ctd_drug, gdsc1_drug, gdsc2_drug):
#     result_drug_gene = pd.DataFrame()
#     for gene in gene_increase['gene']:
#         gene_file = pd.read_csv(os.path.join(current_directory, 'static', 'file', 'gene', '{}.csv'.format(gene)))
#         gene_file = gene_file[((gene_file['source'] == "primary") & gene_file['drug'].isin(primary_drug) |
#                                (gene_file['source'] == "secondary") & gene_file['drug'].isin(secondary_drug) |
#                                (gene_file['source'] == "ctd") & gene_file['drug'].isin(ctd_drug) |
#                                (gene_file['source'] == "gdsc1") & gene_file['drug'].isin(gdsc1_drug) |
#                                (gene_file['source'] == "gdsc2") & gene_file['drug'].isin(gdsc2_drug)) &
#                               (gene_file['correlation'] > 0.4) & (gene_file['p_value'] < 0.05)]
#         result_drug_gene = pd.concat([result_drug_gene, gene_file])
#     result_drug_gene = result_drug_gene.dropna()
#     return result_drug_gene


def actual_candidate(actual, gene_increase, primary_drug, secondary_drug, ctd_drug, gdsc1_drug, gdsc2_drug):
    actual_filtered = actual[actual['gene'].isin(gene_increase['gene'])]
    
    if actual_filtered.empty:
        actual_recommend = pd.DataFrame(columns=['gene', 'drug', 'correlation', 'p_value', 'cal_num', 'source'])
    else:
        # Extract drugs and filter
        actual_drug = actual_filtered['drug'].unique()
        actual_drug1 = [drug for drug in actual_drug if drug in primary_drug or
                        drug in secondary_drug or
                        drug in ctd_drug or
                        drug in gdsc1_drug or
                        drug in gdsc2_drug]

        # Create recommendation DataFrame based on filtered results
        if not actual_drug1:
            actual_recommend = pd.DataFrame(columns=['gene', 'drug', 'correlation', 'p_value', 'cal_num', 'source'])
        else:
            actual_recommend = actual_filtered[actual_filtered['drug'].isin(actual_drug1)].copy()
            actual_recommend['p_value'] = "NA"
            actual_recommend['cal_num'] = "NA"
            actual_recommend['source'] = "actual"
    return actual_recommend


def brand_candidate(brand, gene_increase, primary_drug, secondary_drug, ctd_drug, gdsc1_drug, gdsc2_drug):
    brand_filtered = brand[brand['gene'].isin(gene_increase['gene'])]
    if brand_filtered.empty:
        brand_recommend = pd.DataFrame(columns=['gene', 'drug', 'correlation', 'p_value', 'cal_num', 'source'])
    else:
        brand_recommend = brand_filtered.copy()
        brand_recommend['correlation'] = "NA"
        brand_recommend['p_value'] = "NA"
        brand_recommend['cal_num'] = "NA"
        brand_recommend['source'] = "drugbank"
    return brand_recommend


def merge_all_info(result_drug_gene, gene_increase, sample_prob, drug_effect, gene_info, actual_recommend, brand_recommend, drug_specific):
    # Predicted
    result_predict = pd.merge(result_drug_gene, gene_increase, on='gene', how='inner')
    result_predict_prob = pd.merge(result_predict, sample_prob, on='gene', how='inner')
    result_predict_prob_filtered = result_predict_prob[
        (result_predict_prob['Weighted_Mean'] > 0.5) & (result_predict_prob['category'] == "essential_gene")]
    result_predict_prob_drug = pd.merge(result_predict_prob_filtered, drug_effect, on=['drug', 'source'], how='inner')
    # Actual
    result_actual = pd.merge(actual_recommend, gene_increase, on='gene', how='inner')
    result_actual_prob = pd.merge(result_actual, sample_prob, on='gene', how='inner')
    result_actual_prob_filtered = result_actual_prob[
        (result_actual_prob['Weighted_Mean'] > 0.5) & (result_actual_prob['category'] == "essential_gene")]
    non_primary_data = drug_effect[drug_effect['source'] != 'primary']
    # Calculate mean drug_effect for each drug in non-primary data
    non_primary_mean = non_primary_data.groupby('drug')['drug_effect'].mean().reset_index()
    # Filter primary data
    primary_data = drug_effect[drug_effect['source'] == 'primary'][['drug', 'drug_effect']].drop_duplicates()
    # Merge primary data with non-primary means
    drug_effect_combined = pd.merge(primary_data, non_primary_mean, on='drug', how='left', suffixes=('_primary', '_non_primary'))
    # Use non-primary mean if available, otherwise use primary data
    drug_effect_combined['drug_effect'] = drug_effect_combined['drug_effect_non_primary'].fillna(drug_effect_combined['drug_effect_primary'])
    # Final merge with result_actual_prob_filtered
    result_actual_prob_drug = pd.merge(result_actual_prob_filtered, drug_effect_combined[['drug', 'drug_effect']], on='drug', how='inner')    
    # Marketed
    result_brand = pd.merge(brand_recommend, gene_increase, on='gene', how='inner')
    result_brand_prob = pd.merge(result_brand, sample_prob, on='gene', how='inner')
    result_brand_prob_filtered = result_brand_prob[
        (result_brand_prob['Weighted_Mean'] > 0.7) & (result_brand_prob['category'] == "essential_gene")]
        
    # Merge predicted and actual drug-gene information
    result_all_prob_drug = pd.concat(
        [result_predict_prob_drug, result_actual_prob_drug, result_brand_prob_filtered], axis=0, ignore_index=True, sort=False).fillna("NA")
    result_all_dat1 = pd.merge(result_all_prob_drug, gene_info, on='gene', how='inner')
    # Merge drug specificity and gene specificity information
    result_all_dat = pd.merge(result_all_dat1, drug_specific, on='drug', how='left').fillna("NA")
    result_all_dat = result_all_dat[result_all_dat['rank'] >= 0]
    return result_all_dat

def process_dataframe(df):
    # Sort by drug and cell, and by affinity in descending order
    sorted_data = df.sort_values(['drug', 'cell', 'affinity'], ascending=[True, True, False])

    # Define function to process each group, ensuring 5 rows per group
    def process_group(group):
        if len(group) >= 5:
            return group.head(5)
        else:
            # If fewer than 5 rows, fill remaining positions with first row
            fill_rows = [group.iloc[0]] * (5 - len(group))
            return pd.concat([group, pd.DataFrame(fill_rows)])

    # Apply function and reset index
    result = sorted_data.groupby(['drug', 'cell']).apply(process_group).reset_index(drop=True)

    # Re-sort and add ranking column
    ranked_data = result.sort_values(['drug', 'cell', 'affinity'], ascending=[True, True, False])
    ranked_data['gene_rank'] = ranked_data.groupby(['drug', 'cell']).cumcount() + 1

    # Reshape data to wide format
    wide_data = ranked_data.pivot(
        index=['drug', 'cell'],
        columns='gene_rank',
        values=['affinity', 'prob', 'exp', 'mean_cor']
    )

    # Rename columns and reset index
    wide_data.columns = [f"{col[0]}_{col[1]}" for col in wide_data.columns.values]
    wide_data = wide_data.reset_index()

    # Replace NA with 0 (optional)
    fill_cols = [col for col in wide_data.columns if col.startswith(('affinity_', 'prob_', 'exp_', 'mean_cor_'))]
    wide_data[fill_cols] = wide_data[fill_cols].fillna(0)

    # Adjust column order (consistent with R code)
    col_order = ['drug', 'cell', 'affinity_1', 'prob_1', 'exp_1', 'mean_cor_1',
                 'affinity_2', 'prob_2', 'exp_2', 'mean_cor_2',
                 'affinity_3', 'prob_3', 'exp_3', 'mean_cor_3',
                 'affinity_4', 'prob_4', 'exp_4', 'mean_cor_4',
                 'affinity_5', 'prob_5', 'exp_5',  'mean_cor_5']
    wide_data = wide_data[col_order]

    return wide_data


# Define attention module
class AttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(AttentionModule, self).__init__()
        # Check if channel count is reasonable
        if in_channels < reduction_ratio:
            reduction_ratio = max(in_channels // 2, 1)
        reduced_channels = in_channels // reduction_ratio

        # Channel Attention
        self.channel_att = nn.Sequential(
            # Parallel average and max pooling
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced_channels, in_channels, kernel_size=1, bias=False)
        )
        
        self.channel_att_max = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Conv1d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced_channels, in_channels, kernel_size=1, bias=False)
        )
        
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.spatial_att = nn.Sequential(
            # Input is concatenation of max and average pooling results along channel dimension
            nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False),  # Use 7x1 convolution kernel
            nn.Sigmoid()
        )

    def forward(self, x):
        # Calculate channel attention
        avg_out = self.channel_att(x)
        max_out = self.channel_att_max(x)
        channel_weight = self.sigmoid_channel(avg_out + max_out)  # Element-wise addition followed by activation
        x = x * channel_weight  # Apply channel attention

        # Calculate spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # Average pooling along channel dimension
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # Max pooling along channel dimension
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)  # Concatenate into 2 channels
        spatial_weight = self.spatial_att(spatial_input)  # Calculate spatial weights
        x = x * spatial_weight  # Apply spatial attention

        return x

# Define residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()

        # Residual path
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)  # Add Dropout layer
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)  # Apply Dropout
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Residual connection
        out = self.relu(out)
        return out
class ResidualCNN(nn.Module):
    def __init__(self, input_channels=1, use_attention=True, n_features=1, 
                 learning_rate=0.01, batch_size=128, lr_patience=3, 
                 lr_factor=0.1, lr_min=1e-06, initial_channels=128, 
                 num_blocks_per_layer=[2, 2, 4], reduction_ratio=4, 
                 dropout_rate=0.2, fc_units=256):
        super(ResidualCNN, self).__init__()
        self.use_attention = use_attention
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_min = lr_min
        self.initial_channels = initial_channels
        self.num_blocks_per_layer = num_blocks_per_layer
        self.reduction_ratio = reduction_ratio
        self.dropout_rate = dropout_rate
        self.fc_units = fc_units
        # Initial convolution layer
        self.conv1 = nn.Conv1d(input_channels, self.initial_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.initial_channels)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        in_channels = self.initial_channels
        self.layers = nn.ModuleList()
        for i, num_blocks in enumerate(self.num_blocks_per_layer):
            out_channels = self.initial_channels * (2 ** i)
            self.layers.append(self._make_layer(in_channels, out_channels, num_blocks=num_blocks, stride=2 if i > 0 else 1,
                                                dropout_rate=self.dropout_rate))
            in_channels = out_channels
            if use_attention:
                setattr(self, f'attn{i + 1}', AttentionModule(out_channels, reduction_ratio=self.reduction_ratio))

        # Pooling layer
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # Calculate fully connected layer input size
        self.n_features = self._calculate_fc_input_size(n_features)

        # Initialize fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.n_features, self.fc_units),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.fc_units, 1)  # Regression task output dimension is 1
        ).to(next(self.parameters()).device)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(in_channels, out_channels, stride, dropout_rate))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if self.use_attention:
                attn = getattr(self, f'attn{i + 1}')
                out = attn(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out.squeeze(-1)  # Remove last dimension

    def _calculate_fc_input_size(self, n_features):
        # Calculate feature size after convolution layers and residual blocks
        size = n_features
        # After layer1 (stride=1)
        # After layer2 (stride=2)
        size = (size - 1) // 2 + 1
        # After layer3 (stride=2)
        size = (size - 1) // 2 + 1
        # After global average pooling
        size = 1

        in_channels = self.layers[-1][-1].conv2.out_channels
        return in_channels * size

# Prediction function
def predict_ic50(test_data):
    # Load test data
    drug_names = test_data.iloc[:, 0].values  # Assume first column is drug names
    test_features = test_data.iloc[:, 2:].values

    # Load saved scaler
    #scaler = joblib.load('/data/cht/new_ic50/test/scaler.pkl')
    with open('/home/www/HDMA/search/static/EXP-DRUG/static/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    test_features = scaler.transform(test_features)

    # Convert to PyTorch tensor and move to device (explicitly specify device)
    X_test = torch.FloatTensor(test_features).to(device)
    X_test = X_test.unsqueeze(1)

    # Create model and move to device
    model = ResidualCNN(input_channels=1, use_attention=True, n_features=X_test.size(2)).to(device)

    # Load model parameters
    model.load_state_dict(torch.load('/home/www/HDMA/search/static/EXP-DRUG/static/trained_model.pth',map_location=device))
    model.eval()

    # Perform prediction
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()
        
    result = pd.DataFrame({
        'drug': drug_names,
        'prediction': y_pred
    })
    return result
def format_significant_figures(df, columns=None, n=4):
    """Format numeric columns using assign() method"""
    # Create formatting function
    def format_value(x):
        return f"{x:.{n}g}" if pd.notna(x) else x
    
    # Create new columns using assign() to avoid chained assignment issues
    result = df.copy()
    for col in columns:
        if col in result.columns:
            result = result.assign(**{col: result[col].apply(format_value)})
    
    return result
     

def action(request):
    if request.method == 'POST':
        gene_data = request.POST.get('gene_data')
        current_directory = "/home/www/HDMA/search/static/EXP-DRUG"

        try:
            user_dat = read_text_data(gene_data)
            user_dat["gene"] = user_dat["gene"].apply(lambda x: x.upper())

            # Create thread pool and process pool
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit I/O-intensive tasks
                future1 = executor.submit(read_process_data, current_directory)
                result1 = future1.result()

                future2 = executor.submit(merge_user_exp, user_dat, result1[2], result1[0])
                rank_dat, correlation, user_exp = future2.result()

                future3 = executor.submit(calcu_ess_num, result1[1], correlation, rank_dat)
                cell_line, cell_cor, cell_pvalue, sample_prob = future3.result()

                future4 = executor.submit(prepare_drug_info, result1[3], result1[4], result1[5], result1[6], result1[7], cell_line)
                primary_drug, secondary_drug, ctd_drug, gdsc1_drug, gdsc2_drug, drug_effect = future4.result()

                future5 = executor.submit(increase_mark, result1[1], cell_line, rank_dat, result1[9])
                gene_increase = future5.result()

            # Submit CPU-intensive tasks
            with concurrent.futures.ProcessPoolExecutor() as executor:
                future6 = executor.submit(drug_candidate, gene_increase, current_directory, primary_drug, secondary_drug, ctd_drug, gdsc1_drug, gdsc2_drug)
                result_drug_gene = future6.result()

            actual_recommend = actual_candidate(result1[-6], gene_increase, primary_drug, secondary_drug, ctd_drug, gdsc1_drug, gdsc2_drug)
            brand_recommend = brand_candidate(result1[-2], gene_increase, primary_drug, secondary_drug, ctd_drug, gdsc1_drug, gdsc2_drug)
            # Merge all information
            result_all_dat = merge_all_info(result_drug_gene, gene_increase, sample_prob, drug_effect, result1[8], actual_recommend, brand_recommend, result1[-4])
            # Remove activated drug-genes
            result_remove_drug = pd.merge(result_all_dat, result1[-5], on=['gene', 'drug'], how='inner')
            result_all_dat = result_all_dat[~result_all_dat['drug'].isin(result_remove_drug['drug'])]
            
            # More target details
            result_details_cor = result_drug_gene[result_drug_gene['drug'].isin(result_all_dat['drug'])]
            result_details_actual = result1[-6][result1[-6]['drug'].isin(result_all_dat['drug'])]
            result_details_brand = result1[-2][result1[-2]['drug'].isin(result_all_dat['drug'])]
            results_details_combind = pd.concat([result_details_cor, result_details_actual, result_details_brand],
                                                axis=0, ignore_index=True, sort=False).fillna("NA")
            
            # Predict sample ic50
            results_merged1 = pd.merge(results_details_combind[['drug', 'gene']], user_exp[['gene', 'exp']], on='gene', how='inner')
            results_merged2 = pd.merge(results_merged1, sample_prob[['gene', 'Weighted_Mean']].rename(columns={'Weighted_Mean': 'prob'}),on='gene', how='inner')
            results_merged3 = pd.merge(results_merged2, result1[-1], on=['drug','gene'], how='inner')
            results_details_combind_new = results_details_combind.replace('NA', np.nan)
            cor_mean = results_details_combind_new.groupby(['drug', 'gene'])['correlation'].mean().reset_index().rename(columns={'correlation': 'mean_cor'})
            results_merged4 = pd.merge(results_merged3, cor_mean, on=['drug','gene'], how='inner')
            results_merged4['cell'] = 'pred_cell'
            results_merged4 = results_merged4.replace('NA', np.nan)
            
            results_merged_trans = process_dataframe(results_merged4)
            results_average_origin = predict_ic50(results_merged_trans)
            result_all_dat = pd.merge(results_average_origin, result_all_dat, on='drug', how='right')
            
            result_all_dat['drug_specific_ratio'] = pd.to_numeric(result_all_dat['drug_specific_ratio'], errors='coerce')
            result_all_dat['prediction'] = pd.to_numeric(result_all_dat['prediction'], errors='coerce')
            result_all_dat['drug_effect'] = pd.to_numeric(result_all_dat['drug_effect'], errors='coerce')
            result_all_dat['Weighted_Mean'] = pd.to_numeric(result_all_dat['Weighted_Mean'], errors='coerce')
            result_all_dat['correlation'] = pd.to_numeric(result_all_dat['correlation'], errors='coerce')
            result_all_dat['lof'] = pd.to_numeric(result_all_dat['lof'], errors='coerce')
            result_all_dat['shet'] = pd.to_numeric(result_all_dat['shet'], errors='coerce')
            
            # Sort and select top 5 drugs
            if (result_all_dat['drug_specific_ratio'] >= 0.6).any():
                # If values > 0.6 exist, perform first operation
                result_all_dat = result_all_dat[
                    (result_all_dat['drug_specific_ratio'] >= 0.6) | pd.isna(result_all_dat['drug_specific_ratio'])]
                result_all_dat = result_all_dat[result_all_dat['prediction'].isin(
                    result_all_dat.sort_values(by='prediction')['prediction'].unique()[:5])]

            else:
                # If no values > 0.6, perform second operation
                result_all_dat = result_all_dat[result_all_dat['prediction'].isin(
                    result_all_dat.sort_values(by='prediction')['prediction'].unique()[:5])]
            
            # Visualize main targets
            formatted_results = format_significant_figures(result_all_dat, columns=['drug_specific_ratio', 'prediction', 'drug_effect', 'Weighted_Mean', 'correlation', 'lof', 'shet'])
            result_all = pd.merge(formatted_results, result1[-3], on=['drug', 'source'], how='left')
            result_all['drug'] = np.where(pd.isna(result_all['database_compound']), result_all['drug'], result_all['database_compound'])
            results = result_all.to_dict(orient='records')

            # More target information visualization
            results_details_combind = results_details_combind[results_details_combind['drug'].isin(result_all_dat['drug'])]
            results_details_gene = pd.merge(results_details_combind, result1[8], on='gene', how='inner').fillna("NA")
            results_details_gene['correlation'] = pd.to_numeric(results_details_gene['correlation'], errors='coerce')
            results_details_gene['lof'] = pd.to_numeric(results_details_gene['lof'], errors='coerce')
            results_details_gene['shet'] = pd.to_numeric(results_details_gene['shet'], errors='coerce')
            formatted_results = format_significant_figures(results_details_gene, columns=['drug_specific_ratio', 'prediction', 'drug_effect', 'Weighted_Mean', 'correlation', 'lof', 'shet'])
            result_details_drug = pd.merge(formatted_results, result1[-3], on=['drug', 'source'], how='left')
            result_details_drug['drug'] = np.where(pd.isna(result_details_drug['database_compound']), result_details_drug['drug'],
                                          result_details_drug['database_compound'])
            results_details = result_details_drug.to_dict(orient='records')

            return render(request, 'action.html', {'result': results, 'result_details': results_details, 'cell_line': cell_line, 'cell_cor': cell_cor, 'cell_pvalue': cell_pvalue})

        except (KeyError, ValueError) as e:

            return HttpResponse("File format error. The file must contain 'gene' and 'exp' columns, and expression levels should not contain missing values.")

    else:
        return HttpResponse("File upload failed")