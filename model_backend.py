import pandas as pd 
import numpy as np 
import xgboost as xgb 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from jinja2 import Environment, FileSystemLoader
import pdfkit
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
import seaborn as sns
from scipy.stats import pearsonr 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
import seaborn as sns
from code_groupings import new_feats
import shap 
import joblib
from datetime import datetime
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
import warnings 
warnings.simplefilter(action='ignore', category=FutureWarning)
from pandas.errors import DtypeWarning
warnings.simplefilter('ignore', DtypeWarning)
import pgeocode
from geopy.distance import geodesic
from tqdm import tqdm
import requests
import networkx as nx
from scipy.special import expit
from url_info import code_urls, drug_urls, MOST_UP_TO_DATE_CMS_YEAR, nppes_url
from model_parameters import XGB_PARAMS
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.inspection import PartialDependenceDisplay

#### GLOBAL VARIABLES ####

IDOS_VAL_COLUMN = 'is_idose'

CMS_CONVERSIONS = {
    'Dorzolamide And Timolol':'Dorzolamide-Timolol',
}


#### HELPER FUNCTIONS ####

def balance_classes(X, y):
    smote = SMOTE()
    X_smote, y_smote = smote.fit_resample(X, y)
    
    return X_smote, y_smote

def plot_importance(clf, importance_type, max_num_features): 
    #importances = clf.get_booster().get_score(importance_type=importance_type)
    importances = dict(zip(clf.feature_names_in_, clf.feature_importances_))
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:max_num_features]
    features, scores = zip(*sorted_importances)
    if max_num_features > len(features):
        max_num_features = len(features)
    
    plt.figure(figsize=(10, 0.4*max_num_features))
    bars = plt.barh(range(len(scores)), scores, color='skyblue')
    
    plt.yticks(range(len(features)), features)
    plt.gca().invert_yaxis()
    
    for i, score in enumerate(scores):
        plt.text(score + max(scores) * 0.01, i, f'{score:.2f}', va='center')
        
    plt.xlabel(f'Importance ({importance_type})', fontsize=12)
    plt.title(f'Top {max_num_features} Features', fontsize=14)
    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.5)

def get_importances(clf, max_num_features):
    #importances = dict(sorted(clf.get_booster().get_score(importance_type='gain').items(), key=lambda item: item[1], reverse=True))
    importances = dict(zip(clf.feature_names_in_, clf.feature_importances_))
    importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    #print(list(importances.items()))
    plot_importance(clf, importance_type='gain', max_num_features=max_num_features)
    plt.savefig('images\\importances.png') 
    plt.close()
    
    contributions = np.array(list(importances.values()))/sum(importances.values())*100
    feature_df = pd.DataFrame({
        'Feature': importances.keys(), 
        'Importance': importances.values(), 
        'Contribution': contributions
    }).sort_values(by="Importance", ascending=False)   
    
    return feature_df

def plot_confusion_matrix(clf, X_val, y_val, path):
    #print(clf)
    y_pred = clf.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    #ConfusionMatrixDisplay.from_estimator(clf, X_val, y_val, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    
def plot_correlation(clf, X_val, y_val, path): 
    y_pred = clf.predict(X_val)
    r, _ = pearsonr(y_val, y_pred)
    
    sns.set(style='whitegrid')
    
    plt.figure(figsize=(8,6))
    sns.regplot(x=y_val, y=y_pred, ci=None, scatter_kws={"s": 30, "alpha": 0.7})
    
    min_val = min(np.min(y_val), np.min(y_pred))
    max_val = min(np.max(y_val), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Ideal Fit")
    
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("True vs Predictions")
    plt.text(0.05, 0.95, f'r = {r:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
   
def plot_xgb_tree_manual(clf, tree_path, tree_index=0, figsize=(30,10)): 
    
    def parse_node_id(x):
        if isinstance(x, (int, float)): 
            return int(x)
        s = str(x)
        if '-' in s: 
            return int(s.split('-',1)[1])
        return int(s)
    
    df = clf.get_booster().trees_to_dataframe()
    tree_df = df[df['Tree'] == tree_index]
    
    G =  nx.DiGraph()
    
    for _, row in tree_df.iterrows():
        node_id = parse_node_id(row['Node'])
        if row['Feature'] == 'Leaf': 
            leaf_value = float(row['Gain'])
            prob = expit(leaf_value)
            pred = int(prob >= 0.5)
            label = f'Leaf\nPred: {pred}\nProb: {prob:.2f}'
        else: 
            split = float(row['Split'])
            label = f'{row["Feature"]}\n< {split:.3f}'
            
        G.add_node(node_id, label=label)
        
    for _, row in tree_df.iterrows(): 
        if row['Feature'] != 'Leaf': 
            parent = parse_node_id(row['Node'])
            yes_id = parse_node_id(row['Yes'])
            no_id = parse_node_id(row['No'])
            G.add_edge(parent, yes_id, label='Yes')
            G.add_edge(parent, no_id, label='No')
    
    pos = hierarchy_pos(G, root=0)
        
    plt.figure(figsize=figsize)
    
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=5000, node_shape='s')
    nx.draw_networkx_edges(G, pos)
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels, font_size=10)
    
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    
    plt.axis('off')
    plt.title(f'XGBoost Tree {tree_index}')
    plt.tight_layout()
    plt.savefig(tree_path)
    plt.close()
    
def hierarchy_pos(
    G, root=None, width=1500.0, vert_gap=0.2, vert_loc=0, xcenter=None, sibling_gap=0.15
):
    if root is None:
        root = list(G.nodes)[0]
    if xcenter is None:
        xcenter = width / 2.0

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def leaf_count(node):
        children = list(G.successors(node))
        if not children:
            return 1
        return sum(leaf_count(child) for child in children)

    def _hierarchy_pos(node, left, right, vert_loc, pos):
        x = (left + right) / 2.0
        pos[node] = (x, vert_loc)
        children = list(G.successors(node))
        if not children:
            return pos

        total_leaves = sum(leaf_count(child) for child in children)
        total_gap = sibling_gap * (len(children)-1) if len(children) > 1 else 0 
        width_available = (right-left) - total_gap
        
        start = left
        for child in children:
            child_leaves = leaf_count(child)
            child_width = (width_available) * (child_leaves / total_leaves)
            child_left = start
            child_right = start + child_width
            pos = _hierarchy_pos(child, child_left, child_right, vert_loc - vert_gap, pos)
            start += child_width + sibling_gap
        return pos

    return _hierarchy_pos(root, 0, width, vert_loc, pos={})

def combine_cols(df, start_year=None, end_year=None): 
    combine_dict = new_feats
    total_processed = 0
    which_processed = []
    
    new_cols = {}
    
    for header, cols in combine_dict.items(): 
        total_processed += len(cols)
        which_processed += cols 
        new_cols[header] = (df[cols].sum(axis=1))
        
        if start_year is not None and end_year is not None: 
            for year in range(int(start_year), int(end_year)+1):
                year_cols = [f'{col} In {year}' for col in cols]
                new_cols[f'{header} In {year}'] = df[year_cols].sum(axis=1)
                total_processed += len(year_cols)
                which_processed += year_cols
    
    new_df = pd.DataFrame(new_cols)
    
    new_df[IDOS_VAL_COLUMN] = df[IDOS_VAL_COLUMN]
    return new_df, total_processed, which_processed

def get_macs(state_dict, mac_dict, phys_list): 
    macs = []
    for phys in phys_list: 
        if phys not in state_dict.keys(): 
            print(f'Physician {phys} not in state dictionary')
            macs.append('Unknown')
        else:
            state = state_dict[phys]
            mac = mac_dict[state]
            macs.append(mac)
    return macs

def calculate_time_features(X, start_year, end_year): 
    #print(X.columns)
    features = X.columns.str.replace(r'_in_20\d{2}','', regex=True).drop_duplicates()
    #features = features.str.replace(' ','_')
    #print(features)
    time_feats = {}
    
    for feature in features: 
        vals_for_feature = np.column_stack([X[f'{feature}_in_{year}'] for year in range(int(start_year), int(end_year)+1)])
        time_feats[f'{feature}_Rate_of_Change'] = (X[f'{feature}_in_{end_year}'] - X[f'{feature}_in_{start_year}'])/(int(end_year)-int(start_year))
        time_feats[f'{feature}_Median'] = np.median(vals_for_feature, axis=1)
        time_feats[f'{feature}_Standard_Deviation'] = np.std(vals_for_feature, axis=1)
        time_feats[f'{feature}_Range'] = np.max(vals_for_feature, axis=1) - np.min(vals_for_feature, axis=1)
        
    time_features = pd.DataFrame(time_feats)   
        
    return time_features

def generate_model_report(mac_dict, features, top_n_features=10, balance_class=False): 
    if not os.path.exists(f'{os.getcwd()}\\images'): 
        os.makedirs(f'{os.getcwd()}\\images')
    if not os.path.exists(f'{os.getcwd()}\\reports'):
        os.makedirs(f'{os.getcwd()}\\reports')
    
    feature_info = {}
    non_year_features = [feat for feat in features if not feat[-4:].isnumeric()]
    for feature in non_year_features: 
        max_split_vals = len(feature.split('_'))
        for i in range(max_split_vals+1): 
            check = '_'.join(feature.split('_')[:i])
            if check in new_feats.keys(): 
                feature_info[feature] = new_feats[check]
        if feature == 'Min_Dist':
            feature_info[feature] = ['Distance to the nearest iDose user'] 
        elif feature == 'Enum_Time': 
            feature_info[feature] = ['Time since NPI enumeration date']
        elif feature == 'Sole_Prop':
            feature_info[feature] = ['If the provider is the sole proprietor of the institution']
        elif feature in ['Veterans', 'Academic', 'Community Hospital', 'Nonprofit Health System', 'For-Profit Health System', 'Private Practice', 'Surgical Center', 'Unknown']: 
            feature_info[feature] = [f'If the institution type is {feature}']
        elif feature in ['CGS','FCSO','NGS','Noridian','Novitas','Palmetto','WPS']: 
            feature_info[feature] = [f'If the provider MAC is {feature}']
    
    web_feature_info = pd.DataFrame({
        'Feature': list(feature_info.keys()),
        'Description': [', '.join(v) if isinstance(v, list) else str(v) for v in feature_info.values()]
    })
            
    metric_vals = []
    feature_tables = []
    prediction_labels_images = []
    feature_importance_images = []
    class_summaries = []
    shap_summary_images = []
    shap_feature_tables = []
    top_feature_vals = []
    shap_force_image_vals = []
    top_example_table_vals = []
    top_example_mean_vals = []
    top_example_max_vals = []
    top_example_min_vals = []
    par_dep_paths = []
    perm_import_paths = []
    tree_paths = []
    macs = []
    
    web_info = {}
    
    for mac in tqdm(mac_dict.keys()):
        clf = mac_dict[mac]['clf']
        X_val = mac_dict[mac]['X_val']
        y_val = mac_dict[mac]['y_val']
        X_full = mac_dict[mac]['X_full']
        y_full = mac_dict[mac]['y_full']
        clf_type = mac_dict[mac]['type']
        
        if IDOS_VAL_COLUMN in X_full.columns: 
            X_full = X_full.drop(IDOS_VAL_COLUMN, axis=1)
                
        #web_info[mac] = {}
        
        if balance_class: 
            if (y_val == 1).sum() >= 8 and (y_val == 0).sum() >= 8:
                X_val, y_val = balance_classes(X_val, y_val)
            X_full, y_full = balance_classes(X_full, y_full)
                
        macs.append(mac)
    
        #### GENERATE METRIC TABLE ####
        
        y_pred = clf.predict(X_val)
        y_proba = clf.predict_proba(X_val)[:, 1] if hasattr(clf, "predict_proba") else None
        
        class_counts = pd.Series(y_full).value_counts().to_dict()
        class_labels = {1:'iDose Users', 0:'Non iDose Users'}
        class_summary = ", ".join([f'{class_labels.get(cl, f"Class {cl}")} ({count})' for cl, count in class_counts.items()])
        
        if clf_type == 'Binary':
            metrics = {
                "Accuracy": round(accuracy_score(y_val, y_pred), 4),
                "Precision": round(precision_score(y_val, y_pred), 4),
                "Recall": round(recall_score(y_val, y_pred), 4),
                "F1 Score": round(f1_score(y_val, y_pred), 4),
                "ROC AUC": round(roc_auc_score(y_val, y_proba), 4) if y_proba is not None else "N/A"
            }  
        elif clf_type == 'Regression':
            metrics = {
                "MAE": round(mean_absolute_error(y_val, y_pred), 4),
                "MSE": round(mean_squared_error(y_val, y_pred), 4),
                "RMSE": round(np.sqrt(mean_squared_error(y_val, y_pred)), 4),
                "R2 Score": round(r2_score(y_val, y_pred), 4)
            }
            
        #### CONFUSION MATRIX ####
        
        cm_path = f"{os.getcwd()}\\images\\confusion_matrix_{mac}.png"
        plot_confusion_matrix(clf, X_val, y_val, cm_path)
        
        #### FEATURE IMPORTANCE ####
        
        clf.fit(X_full, y_full)
        feature_df = get_importances(clf, top_n_features)
        if feature_df is not None: 
            fi_path = f'{os.getcwd()}\\images\\importances_{mac}.png'
        
        #### SHAP ANALYSIS ####
        
        X_full = X_full.astype(float)
        explainer = shap.TreeExplainer(clf, X_full, model_output='probability')
        shap_values = explainer(X_full)
        shap_matrix = shap_values.values
        
        shap.summary_plot(shap_values, X_full, max_display=X_full.shape[1], show=False)
        shap_summary_filename = f'{os.getcwd()}\\images\\shap_summary_{mac}.png'
        plt.savefig(shap_summary_filename, bbox_inches='tight')
        plt.close()
            
        shap_df = pd.DataFrame(shap_values.values, columns=X_full.columns)
        mean_shap = shap_df.abs().mean().sort_values(ascending=False)
        shap_importance = pd.Series(mean_shap, index=X_full.columns).sort_values(ascending=False)
        xgb_importance = pd.Series(clf.feature_importances_, index=X_full.columns).sort_values(ascending=False)
        importance_df = pd.DataFrame({
            'Feature': xgb_importance.index.sort_values(),
            'SHAP': shap_importance,
            'XGB': xgb_importance
        })
        
        feature_df = pd.merge(feature_df, importance_df.drop('XGB',axis=1), on='Feature').sort_values(by='SHAP', ascending=False)
        
        top_features = mean_shap.head(3).index.tolist()
        top_example_tables = {} 
        top_example_means = {}
        top_example_maxes = {}
        top_example_mins = {}
        
        for feature in top_features: 
            i = X_full.columns.get_loc(feature)
            shap_vals = shap_matrix[:, i]
            top_indices = np.argsort(np.abs(shap_vals))[::-1][:10]
            top_example_means[feature] = np.mean(X_full[feature])
            top_example_maxes[feature] = np.max(X_full[feature])
            top_example_mins[feature] = np.min(X_full[feature])
            
            top_examples = []
            for idx in top_indices: 
                top_examples.append({
                    "NPI": X_full.index[idx],
                    "SHAP": shap_vals[idx],
                    "FeatureValue": X_full.iloc[idx, i],
                    "Prediction": clf.predict_proba(X_full)[idx, 1]
                })
                
            top_example_tables[feature] = top_examples
        
        shap_force_images = {}
        for feature in top_features: 
            i = X_full.columns.get_loc(feature)
            shap_vals = shap_matrix[:, i]
            top_idx = np.argmax(np.abs(shap_vals))
            
            shap.plots.waterfall(shap_values[top_idx], show=False)
            filename = f'{os.getcwd()}\\images\\shap_force_{feature}_{mac}.png'
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            
            shap_force_images[feature] = filename
            
        shap_feature_table = mean_shap.reset_index()
        shap_feature_table.columns = ['Feature', 'Mean_SHAP']
        shap_feature_table = shap_feature_table.loc[shap_feature_table['Feature'].isin(top_features)]
        shap_feature_table = shap_feature_table.to_dict(orient='records')
        
        #### PARTIAL DEPENDENCE PLOT ###
        
        temp_clf = GradientBoostingClassifier(**XGB_PARAMS)
        temp_clf.fit(X_full, y_full)
        new_import = pd.DataFrame({
            'Feature': X_full.columns, 
            'Importance': temp_clf.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        top_six = new_import['Feature'].to_list()[:6]

        fig, axs = plt.subplots(3, 2, figsize=(15,15))
        
        PartialDependenceDisplay.from_estimator(temp_clf, X_full, features=top_six, ax=axs, response_method='predict_proba', method='brute')
        plt.tight_layout()
        par_dep_path = f'{os.getcwd()}\\images\\partial_dependence_{mac}.png'
        plt.savefig(par_dep_path)
        plt.close() 
        
        #### PERMUTATION IMPORTANCE ####
        
        results = permutation_importance(clf, X_full, y_full, n_repeats=10)
        importances = results.importances_mean 
        std = results.importances_std 
        indices = importances.argsort()[::-1]
        feature_names = X_full.columns[indices]
        
        plt.figure(figsize=(8,6))
        plt.barh(feature_names, importances[indices], xerr=std[indices], align='center')
        plt.xlabel('Mean Decrease in Accuracy')
        plt.title('Permutation Importance')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        perm_import_path = f'{os.getcwd()}\\images\\permutation_importance_{mac}.png'
        plt.savefig(perm_import_path)
        plt.close()
        
        #### TREE VISUALIZATION ####
        
        # fig, ax = plt.subplots(figsize=(20,10), dpi=300)
        # plot_tree(clf, num_trees=0, rankdir='UT', ax=ax)
        tree_path = f'{os.getcwd()}\\images\\xgb_tree_{mac}.png'
        #plot_xgb_tree_manual(clf, tree_path, 0, (50, 10))
        
        prediction_labels_images.append(cm_path)
        feature_importance_images.append(fi_path)        
        shap_summary_images.append(shap_summary_filename)
        shap_feature_tables.append(shap_feature_table)
        metric_vals.append(metrics)
        class_summaries.append(class_summary)
        tree_paths.append(tree_path)
        feature_tables.append(feature_df.to_dict(orient="records"))
        top_feature_vals.append(top_features)
        shap_force_image_vals.append(shap_force_images)
        top_example_table_vals.append(top_example_tables)
        top_example_mean_vals.append(top_example_means)
        top_example_max_vals.append(top_example_maxes)
        top_example_min_vals.append(top_example_mins)
        par_dep_paths.append(par_dep_path)
        perm_import_paths.append(perm_import_path)
        
        web_info[mac] = {
            'CLASS_SUMMARY':class_summary,
            'METRICS':metrics,
            'FEATURE_INFO':web_feature_info,
            'CM_IMAGES':cm_path,
            'SHAP_SUMMARIES':shap_summary_filename,
            'PAR_DEP':par_dep_path
        }
        
    
    #### OUTPUT PDF REPORT ####
        
    env = Environment(loader=FileSystemLoader("html_templates"))
    template = env.get_template("report_template.html")
    html_content = template.render(
        metric_vals=metric_vals, 
        feature_tables=feature_tables,
        prediction_labels_images=prediction_labels_images,
        feature_importance_images=feature_importance_images, 
        class_summaries=class_summaries, 
        shap_summary_images=shap_summary_images,
        shap_feature_tables=shap_feature_tables,
        top_feature_vals=top_feature_vals,
        shap_force_image_vals=shap_force_image_vals,
        top_example_table_vals=top_example_table_vals,
        top_example_mean_vals=top_example_mean_vals, 
        top_example_max_vals=top_example_max_vals,
        top_example_min_vals=top_example_min_vals, 
        par_dep_paths=par_dep_paths, 
        perm_import_paths=perm_import_paths, 
        tree_paths=tree_paths, 
        macs=macs,
        feature_info=feature_info
    )

    import shutil
    #path_wkhtmltopdf = 'wkhtmltopdf.exe' if os.path.exists('wkhtmltopdf.exe') else r'"/usr/bin/wkhtmltopdf"'
    #path_wkhtmltopdf = 'wkhtmltopdf.exe'
    
    path_wkhtmltopdf = shutil.which('wkhtmltopdf')
    config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
    options = {'enable-local-file-access': None}

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")  
    pdf_file = f'reports\\model_report_{formatted_datetime}.pdf'  
    pdfkit.from_string(html_content, pdf_file, configuration=config, options=options)
    #webbrowser.open_new_tab(f'{os.getcwd()}\\{pdf_file}')
    
    return pdf_file, web_info


def calculate_min_distance(npi_zip_df, idose_zips, progress_updater=None, idx=1):      
    zips = npi_zip_df.to_list()
    
    
    nomi = pgeocode.Nominatim('us')
    min_dists = []
    
    zip_coords = {}
    for zipcode in tqdm(zips, desc='Precomputing Zips'): 
        res = nomi.query_postal_code(zipcode)
        if res is not None and not np.isnan(res.latitude): 
            zip_coords[zipcode] = (res.latitude, res.longitude)
    for zipcode in idose_zips: 
        if zipcode in zip_coords.keys():
            continue
        res = nomi.query_postal_code(zipcode)
        if res is not None and not np.isnan(res.latitude): 
            zip_coords[zipcode] = (res.latitude, res.longitude)
    
    for i, zip1 in tqdm(enumerate(zips), desc='Calculating Locations', total=len(zips)):
        min_dist = np.inf
        coord1 = zip_coords[zip1]
        
        for j, zip2 in enumerate(idose_zips): 
            if i == j: 
                continue 
                    
            coord2 = zip_coords[zip2]
                        
            distance_km = geodesic(coord1, coord2).km 
            #print(f'Dist from {zip1} to {zip2} is {distance_km}')
            if distance_km < min_dist: 
                min_dist = distance_km
        
        if progress_updater: 
            progress_updater(idx, f'Computing nearest iDose distance for {npi_zip_df.index[i]}')
            idx += 1
            
        min_dists.append(min_dist)
    
    min_dist_df = pd.DataFrame({'Min_Dist':min_dists}, index=npi_zip_df.index)
    return min_dist_df  



#### MAIN FUNCTIONS ####



def format_cms_data(df, start_year, end_year, idose_zips):
    #print(df[['DRUG_Combigan+Brimonidine Tartrate/Timolol_Total_Services', 'DRUG_Dorzolamide-Timolol+Dorzolamide Hcl/Timolol Maleat_Total_Services']])
    df = df.set_index('NPI') 
    other_cols = []
    for column in df.columns: 
        new_column = column
        vals = column.split('_')
        if len(vals) == 5: 
            new_column = f'{vals[3]}_{vals[1]}_in_{vals[4]}'
        elif len(vals) == 4:
            new_column = f'{vals[3]}_{vals[1]}'
        else:
            other_cols.append(column)
        
        df = df.rename(columns={column:new_column})
    
    new_cols = {}
    for header, cols in new_feats.items(): 
        bene_vals = [col for col in df.columns if any(val in col for val in cols) and 'Beneficiaries_' in col and '_20' in col]
        serv_vals = [col for col in df.columns if any(val in col for val in cols) and 'Services_' in col and '_20' in col]
        
        if header == 'Brimonidine' and any('Combigan' in bene for bene in bene_vals): 
            bene_vals = [val for val in bene_vals if 'Combigan' not in bene_vals]
            serv_vals = [val for val in serv_vals if 'Combigan' not in serv_vals]
        if header == 'Timolol' and any('Dorzolamide' in bene for bene in bene_vals) or any('Combigan' in bene for bene in bene_vals): 
            bene_vals = [val for val in bene_vals if 'Dorzolamide' not in val and 'Combigan' not in val]
            serv_vals = [val for val in serv_vals if 'Dorzolamide' not in val and 'Combigan' not in val]
        
        if len(bene_vals) > 0 and len(serv_vals) > 0:
            new_cols[f'{header}_Beneficiaries_TOTAL'] = (df[bene_vals].sum(axis=1))
            new_cols[f'{header}_Services_TOTAL'] = (df[serv_vals].sum(axis=1))
            
            if start_year != end_year: 
                for year in range(int(start_year), int(end_year)+1): 
                    #year_cols = [f'{col}_in_{year}' for col in cols]
                    
                    bene_year_cols = [col for col in df.columns if any(val in col for val in cols) and 'Beneficiaries' in col and str(year) in col]
                    srvcs_year_cols = [col for col in df.columns if any(val in col for val in cols) and 'Services' in col and str(year) in col]
                    
                    if header == 'Brimonidine' and any('Combigan' in bene for bene in bene_year_cols): 
                        bene_year_cols = [val for val in bene_year_cols if 'Combigan' not in bene_year_cols]
                        srvcs_year_cols = [val for val in srvcs_year_cols if 'Combigan' not in srvcs_year_cols]
                    if header == 'Timolol' and any('Dorzolamide' in bene for bene in bene_year_cols) or any('Combigan' in bene for bene in bene_year_cols): 
                        bene_year_cols = [val for val in bene_year_cols if 'Dorzolamide' not in val and 'Combigan' not in val]
                        srvcs_year_cols = [val for val in srvcs_year_cols if 'Dorzolamide' not in val and 'Combigan' not in val]

                    new_cols[f'{header}_Beneficiaries_in_{year}'] = df[bene_year_cols].sum(axis=1)
                    new_cols[f'{header}_Services_in_{year}'] = df[srvcs_year_cols].sum(axis=1)
    
    val_df = pd.DataFrame(new_cols, index=df.index)
    final_df = pd.merge(val_df, df[other_cols], left_index=True, right_index=True)
    
    min_dist_df = calculate_min_distance(final_df['ZIP'], idose_zips)
    final_df = pd.merge(final_df, min_dist_df, left_index=True, right_index=True).drop(['State','ZIP'], axis=1)
    
    now = datetime.now() 
    times = [(now - datetime.strptime(date, "%Y-%m-%d")).days for date in final_df['Enum_Date'].to_list()]
    final_df['Enum_Time'] = times 
    final_df = final_df.drop('Enum_Date', axis=1)

    if str(start_year) != str(end_year):
        time_df = final_df.loc[:, final_df.columns.str.contains('in_20')]
        final_df = pd.concat([final_df, calculate_time_features(time_df, start_year, end_year)], axis=1)
    else:    
        final_df = final_df.drop([col for col in final_df.columns if MOST_UP_TO_DATE_CMS_YEAR in col], axis=1)
    
    for year in range(int(start_year), int(end_year)+1):
        for col in final_df.columns: 
            if str(year) in col: 
                final_df = final_df.drop(col, axis=1)
    
    return final_df[~final_df.index.duplicated()]


def format_uploaded_data(df, start_year, end_year, idose_zips, progress_updater=None): 
        
    new_cols = {}
    df = df.set_index('NPI').drop('Name', axis=1).replace('<11', 5).astype(int)
    df.columns = df.columns.str.replace(' Patients','', regex=False)
    df.columns = df.columns.str.replace(' Claims','', regex=False)
    
    idx = 1
    if progress_updater: 
        progress_updater(idx, f'Combining columns based on feature codes...')
        idx+=1
    
    # for key, value in CMS_CONVERSIONS.items():
    #     df.columns = df.columns.str.replace(key, value, regex=False)
        
    for header, cols in new_feats.items(): 
        new_cols[header] = (df[cols].sum(axis=1))
        
        if start_year != MOST_UP_TO_DATE_CMS_YEAR: 
            for year in range(int(start_year), int(end_year)+1):
                year_cols = [f'{col} In {year}' for col in cols]
                new_cols[f'{header} In {year}'] = df[year_cols].sum(axis=1)
        
    new_df = pd.DataFrame(new_cols)
    new_df.index = df.index
        
    final_df = new_df.loc[:, ~new_df.columns.str.contains(' 20')].astype(float)
    nppes_df = get_nppes_uploads(final_df.index, progress_updater, idx).set_index('NPI')
    final_df = pd.merge(final_df, nppes_df, left_index=True, right_index=True)

    idx += len(final_df)
                            
    if progress_updater: 
        progress_updater(idx, f'Getting MAC information...')
        idx+=1
                        
    state_to_mac = pd.read_csv('state_to_mac.csv').set_index('State')['MAC'].to_dict()
    final_df['MAC'] = [state_to_mac[state] for state in final_df['State']]
    
    min_dist_df = calculate_min_distance(final_df['ZIP'], idose_zips, progress_updater, idx)
    final_df = pd.merge(final_df, min_dist_df, left_index=True, right_index=True).drop(['State','ZIP'], axis=1)
    idx += len(final_df)

    if progress_updater: 
        progress_updater(idx, f'Getting NPI enumeration dates...')
        idx+=1 

    now = datetime.now() 
    times = [(now - datetime.strptime(date, "%Y-%m-%d")).days for date in final_df['Enum_Date'].to_list()]
    final_df['Enum_Time'] = times 
    final_df = final_df.drop('Enum_Date', axis=1)

    if progress_updater: 
        progress_updater(idx, f'Computing time features (if applicable)...')
        idx+=1
        
    if str(start_year) != str(end_year):
        time_df = new_df.loc[:, new_df.columns.str.contains('in_20')]
        final_df = pd.concat([final_df, calculate_time_features(time_df, start_year, end_year)], axis=1)
    else:    
        final_df = final_df.drop([col for col in final_df.columns if MOST_UP_TO_DATE_CMS_YEAR in col], axis=1)
        
    for year in range(int(start_year), int(end_year)+1):
        for col in final_df.columns: 
            if str(year) in col: 
                final_df = final_df.drop(col, axis=1)
                
    return final_df


# test_df = pd.DataFrame({
#     'NPI':['1073582409','1083148274','1124314067'],
#     'CPT_92002_Total_Services':[0,1,3],
#     'CPT_92004_Total_Services':[0,1,3],
#     'CPT_92002_Total_Beneficiaries':[0,2,1],
#     'CPT_92004_Total_Beneficiaries':[0,2,1],
#     'DRUG_Vyzulta+Latanoprostene Bunod_Total_Beneficiaries':[45,45,45],
#     'DRUG_Travoprost+Travoprost_Total_Beneficiaries':[45,45,45],
#     'Sole_Prop':['NO','YES','NO'],
#     'is_idose':[True,True,False],
#     'State':['UT','OK','MN'],
#     'ZIP':[84003,84790,84606]
#     })

#format_cms_data(test_df, 2021, 2023)

def get_code_data_from_cms(npi_list, cpt_codes, start_year=MOST_UP_TO_DATE_CMS_YEAR, progress_callback=None, start_idx=0): 
    if int(start_year) < min([int(year) for year in code_urls.keys()]):
        print(f'Start year must be after {min([int(year) for year in code_urls.keys()])}')
        exit(1)
        
    final_df = pd.DataFrame()
    missing_npis = []
    idx = start_idx
    for npi in tqdm(npi_list, total=len(npi_list)): 
        phys_df = pd.DataFrame(columns=['NPI','HCPCS_Cd','State'])
        for year in range(int(start_year),2024):
            BASE_URL = code_urls[str(year)]
            BASE_URL = BASE_URL + f"?filter[Rndrng_NPI]={npi}"
            response = requests.get(BASE_URL)
            response.raise_for_status()
            json_data = response.json()
            if len(json_data) == 0: 
                print(f'NPI {npi} not found in CMS database for {year}')
                continue
            npi_df = pd.DataFrame(json_data)[['Rndrng_NPI','HCPCS_Cd','Tot_Srvcs','Tot_Benes','Rndrng_Prvdr_State_Abrvtn','Rndrng_Prvdr_Zip5']].rename(columns={'Rndrng_NPI':'NPI','Rndrng_Prvdr_State_Abrvtn':'State','Rndrng_Prvdr_Zip5':'ZIP'})
            out_df = npi_df[npi_df['HCPCS_Cd'].isin(cpt_codes)]
            out_df.loc[:,'Tot_Srvcs'] = out_df['Tot_Srvcs'].fillna(0)
            out_df.loc[:,'Tot_Benes'] = out_df['Tot_Benes'].fillna(0)
            out_df.loc[:,'Tot_Srvcs'] = out_df['Tot_Srvcs'].round().astype(int)
            out_df.loc[:,'Tot_Benes'] = out_df['Tot_Benes'].round().astype(int)
            
            out_df = out_df.groupby(['NPI','HCPCS_Cd','State','ZIP']).agg(
                Total_Services=pd.NamedAgg(column='Tot_Srvcs', aggfunc='sum'),
                Total_Beneficiaries=pd.NamedAgg(column='Tot_Benes', aggfunc='sum')
            ).reset_index()

            out_df = out_df.rename(columns={'Total_Services':f'Total_Services_{year}', 'Total_Beneficiaries':f'Total_Beneficiaries_{year}'})

            if phys_df.empty:
                phys_df = out_df
            else:
                phys_df = phys_df.merge(out_df, on=['NPI','HCPCS_Cd','State','ZIP'], how='outer')
                
            #phys_df = phys_df.drop_duplicates('NPI')
           
        if phys_df.empty: 
            missing_npis.append(npi)
                
        final_df = pd.concat([final_df, phys_df]).reset_index(drop=True)
        
        if progress_callback: 
            progress_callback(idx+1, f'Getting CPT Code Info For {npi}...')
            idx += 1
       
    bene_cols = [col for col in final_df.columns if 'Beneficiaries_' in col]
    srvcs_cols = [col for col in final_df.columns if 'Services_' in col]

    new_df = final_df.fillna(0)
    new_df['Total_Beneficiaries'] = new_df[bene_cols].sum(axis=1)
    new_df['Total_Services'] = new_df[srvcs_cols].sum(axis=1)
    
    state_df = new_df[['NPI','State','ZIP']]

    srvcs_pivot = new_df.pivot_table(index='NPI',
                                    columns='HCPCS_Cd',
                                    values=['Total_Services'] + srvcs_cols,
                                    aggfunc='sum',
                                    fill_value=0)

    bene_pivot = new_df.pivot_table(index='NPI',
                                    columns='HCPCS_Cd',
                                    values=['Total_Beneficiaries'] + bene_cols,
                                    aggfunc='sum',
                                    fill_value=0)


    srvcs_pivot.columns = [f"CPT_{c[1]}_{c[0]}" for c in srvcs_pivot.columns]
    bene_pivot.columns = [f"CPT_{c[1]}_{c[0]}" for c in bene_pivot.columns]

    df_final = pd.concat([srvcs_pivot, bene_pivot], axis=1).reset_index() 
    df_final = pd.merge(df_final, state_df, on=['NPI'])
    
    if os.path.exists('state_to_mac.csv'): 
        mac_df = pd.read_csv('state_to_mac.csv').set_index('State')['MAC'].to_dict()
        macs = []
        for state in df_final['State']: 
            mac = mac_df[state]
            macs.append(mac)
        df_final['MAC'] = macs
    
    if start_year == MOST_UP_TO_DATE_CMS_YEAR:
        df_final = df_final.drop([col for col in df_final.columns if MOST_UP_TO_DATE_CMS_YEAR not in col and col not in ['NPI','State','ZIP','MAC']], axis=1)
     
    return df_final.drop_duplicates().reset_index(drop=True), missing_npis


def get_drug_data_from_cms(npi_list, drugs, start_year=MOST_UP_TO_DATE_CMS_YEAR, progress_callback=None, start_idx=0): 
    if int(start_year) < min([int(year) for year in drug_urls.keys()]):
        print(f'Start year must be after {min([int(year) for year in drug_urls.keys()])}')
        exit(1)
    
    total_npis = len(npi_list)
    idx = start_idx
    
    final_df = pd.DataFrame()
    missing_npis = []
    for npi in tqdm(npi_list, total=len(npi_list)): 
        phys_df = pd.DataFrame(columns=['NPI','Prescription'])
        for year in range(int(start_year),2024):
            BASE_URL = drug_urls[str(year)]
            BASE_URL = BASE_URL + f"?filter[Prscrbr_NPI]={npi}"
            response = requests.get(BASE_URL)
            response.raise_for_status()
            json_data = response.json()
            if len(json_data) == 0: 
                print(f'NPI {npi} not found in CMS database for {year}')
                continue
            
            #print(json_data)
            out_df = pd.DataFrame(json_data)[['Prscrbr_NPI','Brnd_Name', 'Gnrc_Name','Tot_Clms','Tot_Benes']].rename(columns={'Prscrbr_NPI':'NPI'})
            out_df = out_df[(out_df['Brnd_Name'].isin(drugs)) | (out_df['Gnrc_Name'].isin(drugs))]
            out_df['Prescription'] = out_df['Brnd_Name'] + '+' + out_df['Gnrc_Name']
            out_df = out_df.drop(['Brnd_Name','Gnrc_Name'], axis=1)
            
            out_df['Tot_Benes'] = pd.to_numeric(out_df['Tot_Benes'], errors='coerce').fillna(0).astype(int)                        
            out_df['Tot_Clms'] = pd.to_numeric(out_df['Tot_Clms'], errors='coerce').fillna(0).astype(int)                                    
            
            out_df = out_df.groupby(['NPI','Prescription']).agg(
                Total_Services=pd.NamedAgg(column='Tot_Clms', aggfunc='sum'),
                Total_Beneficiaries=pd.NamedAgg(column='Tot_Benes', aggfunc='sum')
            ).reset_index()

            out_df = out_df.rename(columns={'Total_Services':f'Total_Services_{year}', 'Total_Beneficiaries':f'Total_Beneficiaries_{year}'})

            #print(phys_df)
            if phys_df.empty:
                phys_df = out_df
            else:
                phys_df = phys_df.merge(out_df, on=['NPI','Prescription'], how='outer')
                
            #phys_df = phys_df.drop_duplicates('NPI')
                   
        if phys_df.empty: 
            missing_npis.append(npi)
           
        final_df = pd.concat([final_df, phys_df]).reset_index(drop=True)
        
        if progress_callback: 
            progress_callback(idx+1, f'Getting Prescription Info For {npi}...')
            idx += 1
       
    bene_cols = [col for col in final_df.columns if 'Beneficiaries_' in col]
    srvcs_cols = [col for col in final_df.columns if 'Services_' in col]

    new_df = final_df.fillna(0)
    new_df['Total_Beneficiaries'] = new_df[bene_cols].sum(axis=1)
    new_df['Total_Services'] = new_df[srvcs_cols].sum(axis=1)

    srvcs_pivot = new_df.pivot_table(index='NPI',
                                    columns='Prescription',
                                    values=['Total_Services'] + srvcs_cols,
                                    aggfunc='sum',
                                    fill_value=0)

    bene_pivot = new_df.pivot_table(index='NPI',
                                    columns='Prescription',
                                    values=['Total_Beneficiaries'] + bene_cols,
                                    aggfunc='sum',
                                    fill_value=0)

    srvcs_pivot.columns = [f"DRUG_{c[1]}_{c[0]}" for c in srvcs_pivot.columns]
    bene_pivot.columns = [f"DRUG_{c[1]}_{c[0]}" for c in bene_pivot.columns]

    df_final = pd.concat([srvcs_pivot, bene_pivot], axis=1).drop_duplicates().reset_index()
    if start_year == MOST_UP_TO_DATE_CMS_YEAR:
        df_final = df_final.drop([col for col in df_final.columns if MOST_UP_TO_DATE_CMS_YEAR not in col and col != 'NPI'], axis=1)
     
    return df_final, missing_npis

def get_nppes_uploads(npi_list, progress_callback=None, start_idx=0):
    final_df = pd.DataFrame()
    missing_npis = []
    idx = start_idx 
    for npi in npi_list: 
        get_url = nppes_url + f'&number={npi}'
        response = requests.get(get_url)
        response.raise_for_status()
        json_data = response.json()  
        
        if not json_data.get('results', []): 
            missing_npis.append(npi)
            continue 
        
        state = json_data['results'][0]['addresses'][0]['state']
        zip_code = json_data['results'][0]['addresses'][0]['postal_code']
        if len(zip_code) == 9: 
            zip_code = zip_code[:-4]
        enum_date = json_data['results'][0]['basic']['enumeration_date']
        sole_prop = json_data['results'][0]['basic']['sole_proprietor']
        
        phys_df = pd.DataFrame({
            'NPI':[npi],
            'State':[state],
            'ZIP':[zip_code],
            'Enum_Date':[enum_date],
            'Sole_Prop':[sole_prop]
        })
        
        final_df = pd.concat([final_df, phys_df]).reset_index(drop=True)
        
        if progress_callback:
            progress_callback(idx, f'Getting NPPES Info for {npi}...')
            idx += 1
            
    return final_df

def get_nppes_info(npi_list, progress_callback=None, start_idx=0): 
    NPPES_URL = nppes_url
    
    final_df = pd.DataFrame()
    missing_npis = []
    idx = start_idx
    for npi in npi_list: 
        get_url = NPPES_URL + f'&number={npi}'
        response = requests.get(get_url)
        response.raise_for_status()
        json_data = response.json()
        
        if not json_data.get('results', []): 
            missing_npis.append(npi)
            continue
        
        sole_prop = json_data['results'][0]['basic']['sole_proprietor']
        enum_date = json_data['results'][0]['basic']['enumeration_date']
        
        phys_df = pd.DataFrame({'NPI':[npi],    
                                'Sole_Prop':[sole_prop],
                                'Enum_Date':[enum_date]
                                })
        
        final_df = pd.concat([final_df, phys_df]).reset_index(drop=True)
        
        if progress_callback: 
            progress_callback(idx+1, f'Getting NPPES Info For {npi}...')
            idx += 1
    
    return final_df, missing_npis

# from itertools import chain
# all_codes = list(chain.from_iterable(new_feats.values()))
# drug_list = [drug for drug in all_codes if drug[-1].isalpha() and drug[0].isalpha()]
# cpt_codes = [code for code in all_codes if not code[-1].isalpha()]
# npi_list = ['1144665878']
# df, miss = get_code_data_from_cms(npi_list, cpt_codes=cpt_codes, start_year='2023')
# #print(df)

# df2, _ = get_drug_data_from_cms(npi_list, drugs=drug_list, start_year='2023')
# #print(df2)
# df3, _ = get_nppes_info(npi_list)
# #print(df3)
# df_temp = pd.merge(df, df2, on=['NPI'])
# all_data = pd.merge(df_temp, df3, on=['NPI'])
# all_data['is_idose'] = True
# print(all_data['CPT_65820_Total_Services_2023'])

# print(format_cms_data(all_data, '2023', MOST_UP_TO_DATE_CMS_YEAR)['GONIOTOMY_Services_TOTAL'])

from miscellaneous import set_cancel_button, set_norm_button

def run_model_mac_split(X, y, balance_class, progress_report, model_name, feat_settings): 
    progress_callback, model_cleaner = progress_report
    
    set_cancel_button()
    cancel_button = st.empty()
    if cancel_button.button('Cancel', key='split_cancel', width='stretch', icon=':material/cancel:'):
        st.stop()
    
    y = y > 0
    
    mac_values = {}
    for mac in np.unique(X['MAC']): 
        mac_values[mac] = {}
        
    mac_clfs = []
    idx = 0
    for mac, mac_df in X.groupby('MAC'): 
        y_mac = y.loc[mac_df.index]
        df = mac_df.drop('MAC', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(df, y_mac, test_size=0.2) 
        
        if balance_class: 
            if len(X_train) > 8:
                X_train, y_train = balance_classes(X_train, y_train)
                df, y_mac = balance_classes(df, y_mac)
        
        # clf = xgb.XGBClassifier(objective='binary:logistic', n_estimators=int(XGB_PARAMS['n_estimators']), subsample=float(XGB_PARAMS['subsample']),
        #                             max_depth=int(XGB_PARAMS['max_depth']), learning_rate=float(XGB_PARAMS['learning_rate']), enable_categorical=True,
        #                             device='cpu', n_jobs=-1, tree_method='hist')
        clf = GradientBoostingClassifier(**XGB_PARAMS)
        clf.fit(X_train, y_train)
     
        
        mac_values[mac]['clf'] = clf
        mac_values[mac]['X_val'] = X_test
        mac_values[mac]['y_val'] = y_test
        mac_values[mac]['X_full'] = df
        mac_values[mac]['y_full'] = y_mac
        mac_values[mac]['type'] = 'Binary'
        
        full_clf = clf 
        full_clf.fit(df, y_mac)
        clf_file_name = f'{model_name}_{mac}.pkl'
        if os.path.exists(clf_file_name) and st.session_state.get('saved_classifiers', []):
            num_extra = 1
            for _, clf_name, _ in st.session_state['saved_classifiers']: 
                if f'{clf_file_name}_overwritten' in clf_name: 
                    num_extra += 1
            os.rename(clf_file_name, f'{clf_file_name}_overwritten{num_extra}')
        
        to_save = {
            'model':full_clf,
            'feat_settings':feat_settings
        }
        joblib.dump(to_save, clf_file_name)
        mac_clfs.append((mac, clf_file_name))
        
        if progress_callback: 
            progress_callback(idx+1, f'Running Model For {mac}...')
            idx += 1
    
    features = X.drop('MAC', axis=1).columns.tolist()
    features = [feature.replace(' Proportion', '') for feature in features]
    features = [feature.replace(' Total', '') for feature in features]
    features = [feature for feature in features if feature not in mac_values.keys()]
    
    model_cleaner()
    
    with st.spinner('Generating reports and analyses...'):
        pdf_file, web_info = generate_model_report(mac_values, features, top_n_features=20, balance_class=balance_class) 
        
    set_norm_button()
    cancel_button.empty()
    
    return mac_clfs, pdf_file, web_info


def run_model_all_macs(X, y, balance_class, model_name, feat_settings): 
    with st.spinner('Running model with this dataset...'):
        st.dataframe(X)
        
        set_cancel_button()
        cancel_button = st.empty()
        if cancel_button.button('Cancel', key='mac_cancel', width='stretch', icon=':material/cancel:'):
            st.stop()
        
        y = y > 0
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
        
        if balance_class: 
            if len(X_train) > 8:
                X_train, y_train = balance_classes(X_train, y_train)
                X, y = balance_classes(X, y)
        
        
        # clf = xgb.XGBClassifier(objective='binary:logistic', n_estimators=int(XGB_PARAMS['n_estimators']), subsample=float(XGB_PARAMS['subsample']),
        #                         max_depth=int(XGB_PARAMS['max_depth']), learning_rate=float(XGB_PARAMS['learning_rate']), enable_categorical=True,
        #                         device='cpu', n_jobs=-1, tree_method='hist')
        
        clf = GradientBoostingClassifier(**XGB_PARAMS)
        
        clf.fit(X_train, y_train)
        
        mac_values = {}
        mac_values['ALL_MACS'] = {}
        mac_values['ALL_MACS']['clf'] = clf 
        mac_values['ALL_MACS']['X_val'] = X_test 
        mac_values['ALL_MACS']['y_val'] = y_test 
        mac_values['ALL_MACS']['X_full'] = X 
        mac_values['ALL_MACS']['y_full'] = y 
        mac_values['ALL_MACS']['type'] = 'Binary' 
        
        full_clf = clf 
        full_clf.fit(X, y)
        clf_file_name = f'{model_name}_ALL_MACS.pkl'
        if os.path.exists(clf_file_name) and st.session_state.get('saved_classifiers', []):
            num_extra = 1
            for _, clf_name, _ in st.session_state['saved_classifiers']: 
                if f'{clf_file_name}_overwritten' in clf_name: 
                    num_extra += 1
            os.rename(clf_file_name, f'{clf_file_name}_overwritten{num_extra}')
            
        to_save = {
            'model':full_clf,
            'feat_settings':feat_settings
        }
        joblib.dump(to_save, clf_file_name)
        
        features = X.columns.tolist()
        features = [feature.replace(' Proportion', '') for feature in features]
        features = [feature.replace(' Total', '') for feature in features]
        # st.text(features)
        # st.text(X.columns.to_list())
    
    with st.spinner('Generating reports and analyses...'):
        pdf_file, web_info = generate_model_report(mac_values, features, top_n_features=20, balance_class=balance_class)
      
    set_norm_button()
    cancel_button.empty()
            
    return clf_file_name, pdf_file, web_info


def train_model(X, y, balance_class, model_name, mac, feat_settings): 
    with st.spinner('Training Model...'): 
        y = y > 0 
        
        if balance_class: 
            if len(X) > 8: 
                X, y = balance_classes(X, y)
        
        # clf = xgb.XGBClassifier(objective='binary:logistic', n_estimators=int(XGB_PARAMS['n_estimators']), subsample=float(XGB_PARAMS['subsample']),
        #                         max_depth=int(XGB_PARAMS['max_depth']), learning_rate=float(XGB_PARAMS['learning_rate']), enable_categorical=True,
        #                         device='cpu', n_jobs=-1, tree_method='hist')
        clf = GradientBoostingClassifier(**XGB_PARAMS)
        
        clf.fit(X, y)
        
        clf_file_name = f'{model_name}_{mac}.pkl'
        if os.path.exists(clf_file_name) and st.session_state.get('saved_classifiers', []):
            num_extra = 1
            for _, clf_name, _ in st.session_state['saved_classifiers']: 
                if f'{clf_file_name}_overwritten' in clf_name: 
                    num_extra += 1
            os.rename(clf_file_name, f'{clf_file_name}_overwritten{num_extra}')
        
        to_save = {
            'model':clf,
            'feat_settings':feat_settings
        }
        joblib.dump(to_save, clf_file_name)
        
        return clf_file_name