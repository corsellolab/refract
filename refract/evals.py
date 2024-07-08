import pandas as pd

def get_substrings_after_first_underscore(strings):
    substrings = [''.join(drug.split('_')[1:]) for drug in strings]
    return substrings
    
def intersection(list1, list2):
    return list(set(list1).intersection(list2))

def recover_valid_biomarkers(ref_df, user_df, k=[5,10,20]):
    #Takes dataframe with biomarker data formatted with columns Compound and Biomarker, and user dataframe with columns Compound and Top Features, 
    #where top features are the 20 strongest features in a list in order of strength. k is a list of top_k which to test recovery on. 
    drugs_with_gt = set(ref_df['Compound'].tolist())
    biomarkers = {}
    for drug in drugs_with_gt:
        df_drug = validated_biomarkers[validated_biomarkers['Compound'] == drug]
        biomarkers[drug] = list(df_drug['Top feature'])
    user_df['Compound'] = user_df['Compound'].apply(lambda x: x.split('_')[0].lower())
    counts = [0] * len(k)
    totals = [0] * len(k)
    for compound in list(user_df['Compound']):
        if compound in drugs_with_gt:
            feats = get_substrings_after_first_underscore(df.loc[df['Compound']==compound]['Top Features'])
            for i in range(len(k)):
                comps = feats[:k[i]]
                overlap = intersection(biomarkers[compound],comps)
                if k[i] < len(biomarkers[compound]):
                    totals[i] = totals[i] + k[i]
                else:
                    totals[i] = totals[i] + len(biomarkers[compound])
                counts[i]  = counts[i] + len(overlap)
    res = []
    for i in range(len(k)):
        res.append(counts[i]/totals[i])
    return res

def compare_results(df1, df2):
    #Gets overlap in output for two different methods.
    df2 = df2.rename(columns={"Top Features" : "Top Features 2"})
    merged = pd.merge(df1,df2,on='Compound')
    merged = merged.dropna()
    top_1 = 0
    top_5 = 0
    top_10 = 0
    top_20 = 0
    for i in range(len(merged)):
        compound1 = merged.iloc[i]["Top Features"]
        compound2 = merged.iloc[i]["Top Features 2"]
        if compound1[0] == compound2[0]:
            top_1 += 1
        top_5 += intersection(compound1[:5],compound2[:5])
        top_10 += intersection(compound1[:5],compound2[:5])
        top_20 += intersection(compound1[:5],compound2[:5])
    return [top_1 / len(merged), top_5 / (5*len(merged)), top_10 / (10*len(merged)), top_20/(20*len(merged))]




