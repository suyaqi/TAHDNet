import pandas as pd

# files can be downloaded from https://mimic.physionet.org/gettingstarted/dbsetup/
med_file = 'prescriptions.csv'
diag_file = 'diagnoses_icd.csv'
procedure_file = 'procedures_icd.csv'

# drug code mapping files (already in ./data/)
ndc2atc_file = 'ndc2atc_level4.csv'
# cid_atc = 'drug-atc.csv'
ndc2rxnorm_file = 'ndc2rxnorm_mapping.txt'

# drug-drug interactions can be down https://www.dropbox.com/s/8os4pd2zmp2jemd/drug-DDI.csv?dl=0
# ddi_file = 'drug-DDI.csv'


def process_procedure():
    pro_pd = pd.read_csv(procedure_file, dtype={'icd9_code': 'category'})
    pro_pd.drop(columns=['row_id'], inplace=True)
    #     pro_pd = pro_pd[pro_pd['SEQ_NUM']<5]
    #     def icd9_tree(x):
    #         if x[0]=='E':
    #             return x[:4]
    #         return x[:3]
    #     pro_pd['ICD9_CODE'] = pro_pd['ICD9_CODE'].map(icd9_tree)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=['subject_id', 'hadm_id', 'seq_num'], inplace=True)
    pro_pd.drop(columns=['seq_num'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)

    return pro_pd


def process_med():
    med_pd = pd.read_csv(med_file, dtype={'ndc': 'category'})
    # filter
    med_pd.drop(columns=['row_id', 'drug_type', 'drug_name_poe', 'drug_name_generic',
                         'formulary_drug_cd', 'gsn', 'prod_strength', 'dose_val_rx',
                         'dose_unit_rx', 'form_val_disp', 'form_unit_disp',
                         'route', 'enddate', 'drug'], axis=1, inplace=True)
    med_pd.drop(index=med_pd[med_pd['ndc'] == '0'].index, axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd['icustay_id'] = med_pd['icustay_id'].astype('int64')
    med_pd['startdate'] = pd.to_datetime(med_pd['startdate'], format='%Y-%m-%d %H:%M:%S')
    med_pd.sort_values(by=['subject_id', 'hadm_id', 'icustay_id', 'startdate'], inplace=True)
    med_pd = med_pd.reset_index(drop=True)

    def filter_first24hour_med(med_pd):
        med_pd_new = med_pd.drop(columns=['ndc'])
        med_pd_new = med_pd_new.groupby(by=['subject_id', 'hadm_id', 'icustay_id']).head([1]).reset_index(drop=True)
        med_pd_new = pd.merge(med_pd_new, med_pd, on=['subject_id', 'hadm_id', 'icustay_id', 'startdate'])
        time_pd = med_pd.drop(columns=['ndc', 'icustay_id']).reset_index(drop=True)
        time_pd = time_pd.sort_values(by=['subject_id', 'hadm_id', 'startdate']).reset_index(drop=True)
        time_pd = time_pd.groupby(by=['subject_id', 'hadm_id']).head([1]).reset_index(drop=True)
        time_pd = time_pd.drop_duplicates()
        time_pd = time_pd.reset_index(drop=True)
        # print(med_pd_new)
        # med_pd_new = med_pd_new.drop(columns=['startdate'])
        # time_pd.to_csv('time_pd.csv')
        return med_pd_new, time_pd

    med_pd, time_pd = filter_first24hour_med(med_pd)
    #     med_pd = med_pd.drop(columns=['STARTDATE'])

    med_pd = med_pd.drop(columns=['icustay_id'])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)

    # visit > 2
    def process_visit_lg2(med_pd):
        a = med_pd[['subject_id', 'hadm_id']].groupby(by='subject_id')['hadm_id'].unique().reset_index()
        a['HADM_ID_Len'] = a['hadm_id'].map(lambda x: len(x))
        a = a[a['HADM_ID_Len'] > 1]
        return a

    med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)
    med_pd = med_pd.merge(med_pd_lg2[['subject_id']], on='subject_id', how='inner')

    return med_pd.reset_index(drop=True), time_pd

def process_diag():
    diag_pd = pd.read_csv(diag_file)
    diag_pd.dropna(inplace=True)
    #     def icd9_tree(x):
    #         if x[0]=='E':
    #             return x[:4]
    #         return x[:3]
    #     diag_pd['ICD9_CODE'] = diag_pd['ICD9_CODE'].map(icd9_tree)
    #     diag_pd = diag_pd[diag_pd['SEQ_NUM'] < 5]
    diag_pd.drop(columns=['seq_num', 'row_id'], inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['subject_id', 'hadm_id'], inplace=True)
    return diag_pd.reset_index(drop=True)


def ndc2atc4(med_pd):
    with open(ndc2rxnorm_file, 'r') as f:
        ndc2rxnorm = eval(f.read())
    med_pd['RXCUI'] = med_pd['ndc'].map(ndc2rxnorm)
    med_pd.dropna(inplace=True)

    rxnorm2atc = pd.read_csv(ndc2atc_file)
    rxnorm2atc = rxnorm2atc.drop(columns=['YEAR', 'MONTH', 'NDC'])
    rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)
    med_pd.drop(index=med_pd[med_pd['RXCUI'].isin([''])].index, axis=0, inplace=True)

    med_pd['RXCUI'] = med_pd['RXCUI'].astype('int64')
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(rxnorm2atc, on=['RXCUI'])
    med_pd.drop(columns=['ndc', 'RXCUI'], inplace=True)
    med_pd = med_pd.rename(columns={'ATC4': 'ndc'})
    med_pd['ndc'] = med_pd['ndc'].map(lambda x: x[:4])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)
    return med_pd


def filter_1000_most_pro(pro_pd):
    pro_count = pro_pd.groupby(by=['icd9_code']).size().reset_index().rename(columns={0: 'count'}).sort_values(
        by=['count'], ascending=False).reset_index(drop=True)
    pro_pd = pro_pd[pro_pd['icd9_code'].isin(pro_count.loc[:1000, 'icd9_code'])]

    return pro_pd.reset_index(drop=True)


def filter_2000_most_diag(diag_pd):
    diag_count = diag_pd.groupby(by=['icd9_code']).size().reset_index().rename(columns={0: 'count'}).sort_values(
        by=['count'], ascending=False).reset_index(drop=True)
    diag_pd = diag_pd[diag_pd['icd9_code'].isin(diag_count.loc[:1999, 'icd9_code'])]

    return diag_pd.reset_index(drop=True)


def filter_300_most_med(med_pd):
    med_count = med_pd.groupby(by=['ndc']).size().reset_index().rename(columns={0: 'count'}).sort_values(by=['count'],
                                                                                                         ascending=False).reset_index(
        drop=True)
    med_pd = med_pd[med_pd['ndc'].isin(med_count.loc[:299, 'ndc'])]

    return med_pd.reset_index(drop=True)


def process_all():
    # get med and diag (visit>=2)
    med_pd, time_pd = process_med()
    med_pd = ndc2atc4(med_pd)
    #     med_pd = filter_300_most_med(med_pd)

    diag_pd = process_diag()
    diag_pd = filter_2000_most_diag(diag_pd)

    pro_pd = process_procedure()
    #     pro_pd = filter_1000_most_pro(pro_pd)

    med_pd_key = med_pd[['subject_id', 'hadm_id']].drop_duplicates()
    diag_pd_key = diag_pd[['subject_id', 'hadm_id']].drop_duplicates()
    pro_pd_key = pro_pd[['subject_id', 'hadm_id']].drop_duplicates()
    time_pd_key = time_pd[['subject_id', 'hadm_id']].drop_duplicates()

    combined_key = med_pd_key.merge(diag_pd_key, on=['subject_id', 'hadm_id'], how='inner')
    combined_key = combined_key.merge(pro_pd_key, on=['subject_id', 'hadm_id'], how='inner')
    combined_key = combined_key.merge(time_pd_key, on=['subject_id', 'hadm_id'], how='inner')

    diag_pd = diag_pd.merge(combined_key, on=['subject_id', 'hadm_id'], how='inner')
    med_pd = med_pd.merge(combined_key, on=['subject_id', 'hadm_id'], how='inner')
    pro_pd = pro_pd.merge(combined_key, on=['subject_id', 'hadm_id'], how='inner')
    time_pd = time_pd.merge(combined_key, on=['subject_id', 'hadm_id'], how='inner')

    # flatten and merge
    diag_pd = diag_pd.groupby(by=['subject_id', 'hadm_id'])['icd9_code'].unique().reset_index()
    med_pd = med_pd.groupby(by=['subject_id', 'hadm_id'])['ndc'].unique().reset_index()
    pro_pd = pro_pd.groupby(by=['subject_id', 'hadm_id'])['icd9_code'].unique().reset_index().rename(
        columns={'icd9_code': 'pro_code'})
    # time_pd = time_pd.groupby(by=['subject_id', 'hadm_id'])['startdate'].unique().reset_index()
    med_pd['ndc'] = med_pd['ndc'].map(lambda x: list(x))
    pro_pd['pro_code'] = pro_pd['pro_code'].map(lambda x: list(x))
    data = diag_pd.merge(med_pd, on=['subject_id', 'hadm_id'], how='inner')
    data = data.merge(pro_pd, on=['subject_id', 'hadm_id'], how='inner')
    data = data.merge(time_pd, on=['subject_id', 'hadm_id'], how='inner')
    #     data['ICD9_CODE_Len'] = data['ICD9_CODE'].map(lambda x: len(x))
    data['NDC_Len'] = data['ndc'].map(lambda x: len(x))
    return data


def statistics():
    print('#patients ', data['subject_id'].unique().shape)
    print('#clinical events ', len(data))

    diag = data['icd9_code'].values
    med = data['ndc'].values
    pro = data['pro_code'].values

    unique_diag = set([j for i in diag for j in list(i)])
    unique_med = set([j for i in med for j in list(i)])
    unique_pro = set([j for i in pro for j in list(i)])

    print('#diagnosis ', len(unique_diag))
    print('#med ', len(unique_med))
    print('#procedure', len(unique_pro))

    avg_diag = 0
    avg_med = 0
    avg_pro = 0
    max_diag = 0
    max_med = 0
    max_pro = 0
    cnt = 0
    max_visit = 0
    avg_visit = 0

    for subject_id in data['subject_id'].unique():
        item_data = data[data['subject_id'] == subject_id]
        x = []
        y = []
        z = []
        visit_cnt = 0
        for index, row in item_data.iterrows():
            visit_cnt += 1
            cnt += 1
            x.extend(list(row['icd9_code']))
            y.extend(list(row['ndc']))
            z.extend(list(row['pro_code']))
        x = set(x)
        y = set(y)
        z = set(z)
        avg_diag += len(x)
        avg_med += len(y)
        avg_pro += len(z)
        avg_visit += visit_cnt
        if len(x) > max_diag:
            max_diag = len(x)
        if len(y) > max_med:
            max_med = len(y)
        if len(z) > max_pro:
            max_pro = len(z)
        if visit_cnt > max_visit:
            max_visit = visit_cnt

    print('#avg of diagnoses ', avg_diag / cnt)
    print('#avg of medicines ', avg_med / cnt)
    print('#avg of procedures ', avg_pro / cnt)
    print('#avg of vists ', avg_visit / len(data['subject_id'].unique()))

    print('#max of diagnoses ', max_diag)
    print('#max of medicines ', max_med)
    print('#max of procedures ', max_pro)
    print('#max of visit ', max_visit)


data = process_all()
statistics()
data.to_pickle('data_gamenet.pkl')
data.head()