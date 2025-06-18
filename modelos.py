import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           precision_recall_curve, roc_curve, auc, accuracy_score,
                           precision_score, recall_score, f1_score)
import xgboost as xgb
import lightgbm as lgb
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import warnings
warnings.filterwarnings('ignore')

# Configuración de gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

# =============================================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# =============================================================================

print("="*80)
print("PROYECTO: CLASIFICACIÓN DE ATTRITION DE EMPLEADOS")
print("="*80)

# Cargar datos
df = pd.read_csv('./data/HR_Analytics.csv')  # Ajusta la ruta según tu archivo
print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# Verificar la variable objetivo
print(f"\nDistribución de la variable objetivo:")
print(df['Attrition'].value_counts())
print(f"Tasa de attrition: {(df['Attrition'] == 'Yes').mean()*100:.2f}%")

# =============================================================================
# 2. FEATURE ENGINEERING Y PREPROCESSING
# =============================================================================

def preprocess_data(df):
    """
    Preprocesa el dataset para modelado
    """
    df_processed = df.copy()
    
    # Convertir variable objetivo a numérica
    df_processed['Attrition'] = (df_processed['Attrition'] == 'Yes').astype(int)
    
    # Eliminar columnas que no aportan información
    columns_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    df_processed = df_processed.drop(columns=columns_to_drop, errors='ignore')
    
    # Label encoding para variables categóricas
    le_dict = {}
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    categorical_columns = categorical_columns.drop('Attrition', errors='ignore')
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        le_dict[col] = le
    
    # Feature engineering adicional
    # Crear nuevas características
    df_processed['Years_Since_Last_Promotion_Ratio'] = (
        df_processed['YearsSinceLastPromotion'] / (df_processed['YearsAtCompany'] + 1)
    )
    
    df_processed['Salary_Per_Year'] = (
        df_processed['MonthlyIncome'] * 12
    )
    
    df_processed['Experience_Company_Ratio'] = (
        df_processed['YearsAtCompany'] / (df_processed['TotalWorkingYears'] + 1)
    )
    
    df_processed['Training_Frequency'] = (
        df_processed['TrainingTimesLastYear'] / 12
    )
    
    # Interacciones importantes
    df_processed['Age_Income_Interaction'] = (
        df_processed['Age'] * df_processed['MonthlyIncome'] / 10000
    )
    
    df_processed['JobLevel_Income_Interaction'] = (
        df_processed['JobLevel'] * df_processed['MonthlyIncome'] / 1000
    )
    
    return df_processed, le_dict

# Preprocesar datos
df_processed, label_encoders = preprocess_data(df)
print(f"\nDatos preprocesados: {df_processed.shape[0]} filas, {df_processed.shape[1]} columnas")
print(f"Nuevas características creadas: {df_processed.shape[1] - df.shape[1]}")

# =============================================================================
# 3. DIVISIÓN DE DATOS (60% Train, 20% Validation, 20% Test)
# =============================================================================

# Separar características y variable objetivo
X = df_processed.drop('Attrition', axis=1)
y = df_processed['Attrition']

print(f"\nForma de X: {X.shape}")
print(f"Forma de y: {y.shape}")

# Primera división: 60% train, 40% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# Segunda división: 20% validation, 20% test del 40% temp
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\nDivisión de datos:")
print(f"Train: {X_train.shape[0]} muestras ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Validation: {X_val.shape[0]} muestras ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Test: {X_test.shape[0]} muestras ({X_test.shape[0]/len(X)*100:.1f}%)")

# Verificar distribución de clases
print(f"\nDistribución de clases:")
print(f"Train - Attrition rate: {y_train.mean()*100:.2f}%")
print(f"Validation - Attrition rate: {y_val.mean()*100:.2f}%")
print(f"Test - Attrition rate: {y_test.mean()*100:.2f}%")

# Escalado de características (solo para Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# 4. DEFINICIÓN DE ESPACIOS DE BÚSQUEDA PARA OPTIMIZACIÓN BAYESIANA
# =============================================================================

# Espacios de búsqueda para cada modelo
search_spaces = {
    'logistic_regression': {
        'C': Real(0.01, 100, prior='log-uniform'),
        'penalty': Categorical(['l1', 'l2']),
        'solver': Categorical(['saga'])
        
    },
    
    'xgboost': {
        'n_estimators': Integer(100, 1000),
        'max_depth': Integer(3, 10),
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'subsample': Real(0.6, 1.0),
        'colsample_bytree': Real(0.6, 1.0),
        'reg_alpha': Real(0, 10),
        'reg_lambda': Real(1, 10)
    },
    
    'lightgbm': {
        'n_estimators': Integer(100, 1000),
        'max_depth': Integer(3, 10),
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'subsample': Real(0.6, 1.0),
        'colsample_bytree': Real(0.6, 1.0),
        'reg_alpha': Real(0, 10),
        'reg_lambda': Real(1, 10),
        'num_leaves': Integer(10, 100)
    }
}

# =============================================================================
# 5. FUNCIÓN PARA ENTRENAR Y EVALUAR MODELOS
# =============================================================================

def train_and_evaluate_model(model, X_train_data, X_val_data, X_test_data, 
                            y_train, y_val, y_test, model_name):
    """
    Entrena un modelo y evalúa su performance
    """
    print(f"\n{'='*60}")
    print(f"ENTRENANDO MODELO: {model_name.upper()}")
    print(f"{'='*60}")
    
    # Entrenar modelo
    model.fit(X_train_data, y_train)
    
    # Predicciones
    y_train_pred = model.predict(X_train_data)
    y_val_pred = model.predict(X_val_data)
    y_test_pred = model.predict(X_test_data)
    
    # Probabilidades para métricas AUC
    if hasattr(model, 'predict_proba'):
        y_train_proba = model.predict_proba(X_train_data)[:, 1]
        y_val_proba = model.predict_proba(X_val_data)[:, 1]
        y_test_proba = model.predict_proba(X_test_data)[:, 1]
    else:
        y_train_proba = model.decision_function(X_train_data)
        y_val_proba = model.decision_function(X_val_data)
        y_test_proba = model.decision_function(X_test_data)
    
    # Calcular métricas
    metrics = {}
    for split, y_true, y_pred, y_proba in [
        ('train', y_train, y_train_pred, y_train_proba),
        ('val', y_val, y_val_pred, y_val_proba),
        ('test', y_test, y_test_pred, y_test_proba)
    ]:
        metrics[split] = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'pr_auc': auc(*precision_recall_curve(y_true, y_proba)[:2])
        }
    
    # Imprimir métricas
    print(f"\nMétricas del modelo {model_name}:")
    print("-" * 60)
    for split in ['train', 'val', 'test']:
        print(f"{split.upper()}:")
        for metric, value in metrics[split].items():
            print(f"  {metric}: {value:.4f}")
        print()
    
    return model, metrics, {
        'train': (y_train, y_train_pred, y_train_proba),
        'val': (y_val, y_val_pred, y_val_proba),
        'test': (y_test, y_test_pred, y_test_proba)
    }

# =============================================================================
# 6. OPTIMIZACIÓN BAYESIANA Y ENTRENAMIENTO DE MODELOS
# =============================================================================

models_results = {}
best_models = {}

# Configuración de validación cruzada
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "="*80)
print("INICIANDO OPTIMIZACIÓN BAYESIANA DE HIPERPARÁMETROS")
print("="*80)

# 1. LOGISTIC REGRESSION
print("\n1. OPTIMIZACIÓN LOGISTIC REGRESSION...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)

# Crear un modelo wrapper para manejar parámetros incompatibles
def lr_objective(**params):
    # Manejar incompatibilidades de parámetros
    if params['penalty'] == 'l1':
        params['solver'] = 'liblinear'
        params.pop('l1_ratio', None)
    elif params['penalty'] == 'l2':
        if params['solver'] == 'saga':
            params['solver'] = 'liblinear'
        params.pop('l1_ratio', None)
    elif params['penalty'] == 'elasticnet':
        params['solver'] = 'saga'
    else:
        params.pop('l1_ratio', None)
    
    model = LogisticRegression(random_state=42, max_iter=1000, **params)
    return model

lr_search = BayesSearchCV(
    lr_model,
    search_spaces['logistic_regression'],
    n_iter=30,
    cv=cv_strategy,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1
)

lr_search.fit(X_train_scaled, y_train)
best_lr = lr_search.best_estimator_
print(f"Mejores parámetros LR: {lr_search.best_params_}")
print(f"Mejor score CV: {lr_search.best_score_:.4f}")

# Evaluar Logistic Regression
lr_model, lr_metrics, lr_predictions = train_and_evaluate_model(
    best_lr, X_train_scaled, X_val_scaled, X_test_scaled, 
    y_train, y_val, y_test, "Logistic Regression"
)

models_results['Logistic Regression'] = {
    'model': lr_model,
    'metrics': lr_metrics,
    'predictions': lr_predictions,
    'best_params': lr_search.best_params_
}

# 2. XGBOOST
print("\n2. OPTIMIZACIÓN XGBOOST...")
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')

xgb_search = BayesSearchCV(
    xgb_model,
    search_spaces['xgboost'],
    n_iter=50,
    cv=cv_strategy,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1
)

xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_
print(f"Mejores parámetros XGB: {xgb_search.best_params_}")
print(f"Mejor score CV: {xgb_search.best_score_:.4f}")

# Evaluar XGBoost
xgb_model, xgb_metrics, xgb_predictions = train_and_evaluate_model(
    best_xgb, X_train, X_val, X_test, 
    y_train, y_val, y_test, "XGBoost"
)

models_results['XGBoost'] = {
    'model': xgb_model,
    'metrics': xgb_metrics,
    'predictions': xgb_predictions,
    'best_params': xgb_search.best_params_
}

# 3. LIGHTGBM
print("\n3. OPTIMIZACIÓN LIGHTGBM...")
lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)

lgb_search = BayesSearchCV(
    lgb_model,
    search_spaces['lightgbm'],
    n_iter=50,
    cv=cv_strategy,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1
)

lgb_search.fit(X_train, y_train)
best_lgb = lgb_search.best_estimator_
print(f"Mejores parámetros LGB: {lgb_search.best_params_}")
print(f"Mejor score CV: {lgb_search.best_score_:.4f}")

# Evaluar LightGBM
lgb_model, lgb_metrics, lgb_predictions = train_and_evaluate_model(
    best_lgb, X_train, X_val, X_test, 
    y_train, y_val, y_test, "LightGBM"
)

models_results['LightGBM'] = {
    'model': lgb_model,
    'metrics': lgb_metrics,
    'predictions': lgb_predictions,
    'best_params': lgb_search.best_params_
}

# =============================================================================
# 7. COMPARACIÓN DE MODELOS
# =============================================================================

print("\n" + "="*80)
print("COMPARACIÓN DE MODELOS")
print("="*80)

# Crear DataFrame de comparación
comparison_data = []
for model_name, results in models_results.items():
    for split in ['train', 'val', 'test']:
        row = {'Model': model_name, 'Split': split}
        row.update(results['metrics'][split])
        comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)

# Mostrar tabla de comparación
print("\nTabla de Comparación de Modelos:")
print("-" * 100)
pivot_table = comparison_df.pivot_table(
    index='Model', 
    columns='Split', 
    values=['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
)
print(pivot_table.round(4))

# Gráfico de comparación de métricas
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']

for i, metric in enumerate(metrics_to_plot):
    row = i // 3
    col = i % 3
    
    # Datos para el gráfico
    metric_data = comparison_df.pivot(index='Model', columns='Split', values=metric)
    
    # Crear gráfico de barras
    x = np.arange(len(metric_data.index))
    width = 0.25
    
    axes[row, col].bar(x - width, metric_data['train'], width, label='Train', alpha=0.8)
    axes[row, col].bar(x, metric_data['val'], width, label='Validation', alpha=0.8)
    axes[row, col].bar(x + width, metric_data['test'], width, label='Test', alpha=0.8)
    
    axes[row, col].set_title(f'{metric.upper()}', fontweight='bold')
    axes[row, col].set_xlabel('Modelos')
    axes[row, col].set_ylabel(metric.capitalize())
    axes[row, col].set_xticks(x)
    axes[row, col].set_xticklabels(metric_data.index, rotation=45)
    axes[row, col].legend()
    axes[row, col].grid(True, alpha=0.3)

plt.suptitle('Comparación de Métricas por Modelo', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# =============================================================================
# 8. VISUALIZACIONES DE VALIDACIÓN
# =============================================================================

def plot_confusion_matrices(models_results, split='test'):
    """
    Crear matrices de confusión para todos los modelos
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (model_name, results) in enumerate(models_results.items()):
        y_true, y_pred, _ = results['predictions'][split]
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   xticklabels=['No Attrition', 'Attrition'],
                   yticklabels=['No Attrition', 'Attrition'])
        axes[i].set_title(f'{model_name}\nConfusion Matrix ({split.capitalize()})', 
                         fontweight='bold')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()

def plot_roc_curves(models_results, split='test'):
    """
    Crear curvas ROC para todos los modelos
    """
    plt.figure(figsize=(10, 8))
    
    for model_name, results in models_results.items():
        y_true, _, y_proba = results['predictions'][split]
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, linewidth=2, 
                label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {split.capitalize()} Set', fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_precision_recall_curves(models_results, split='test'):
    """
    Crear curvas Precision-Recall para todos los modelos
    """
    plt.figure(figsize=(10, 8))
    
    for model_name, results in models_results.items():
        y_true, _, y_proba = results['predictions'][split]
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, linewidth=2,
                label=f'{model_name} (AUC = {pr_auc:.3f})')
    
    # Línea de referencia (baseline)
    baseline = y_true.mean() if split == 'test' else y_test.mean()
    plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1,
               label=f'Baseline ({baseline:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curves - {split.capitalize()} Set', fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_feature_importance(models_results):
    """
    Mostrar importancia de características para modelos tree-based
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    tree_models = ['XGBoost', 'LightGBM']
    feature_names = X.columns
    
    for i, model_name in enumerate(tree_models):
        if model_name in models_results:
            model = models_results[model_name]['model']
            importance = model.feature_importances_
            
            # Ordenar por importancia
            indices = np.argsort(importance)[::-1][:20]  # Top 20
            
            axes[i].barh(range(len(indices)), importance[indices])
            axes[i].set_yticks(range(len(indices)))
            axes[i].set_yticklabels([feature_names[j] for j in indices])
            axes[i].set_title(f'Top 20 Feature Importance - {model_name}', fontweight='bold')
            axes[i].set_xlabel('Importance')
            axes[i].invert_yaxis()
    
    plt.tight_layout()
    plt.show()

# Generar todas las visualizaciones
print("\n" + "="*80)
print("GENERANDO VISUALIZACIONES DE VALIDACIÓN")
print("="*80)

# Matrices de confusión
print("\n1. Matrices de Confusión (Test Set):")
plot_confusion_matrices(models_results, 'test')

# Curvas ROC
print("\n2. Curvas ROC (Test Set):")
plot_roc_curves(models_results, 'test')

# Curvas Precision-Recall
print("\n3. Curvas Precision-Recall (Test Set):")
plot_precision_recall_curves(models_results, 'test')

# Importancia de características
print("\n4. Importancia de Características:")
plot_feature_importance(models_results)

# =============================================================================
# 9. ANÁLISIS FINAL Y SELECCIÓN DEL MEJOR MODELO
# =============================================================================

print("\n" + "="*80)
print("ANÁLISIS FINAL Y SELECCIÓN DEL MEJOR MODELO")
print("="*80)

# Calcular scores compuestos para ranking
def calculate_composite_score(metrics, weights=None):
    """
    Calcula un score compuesto basado en múltiples métricas
    """
    if weights is None:
        weights = {'roc_auc': 0.3, 'pr_auc': 0.3, 'f1': 0.25, 'precision': 0.15}
    
    score = 0
    for metric, weight in weights.items():
        score += metrics[metric] * weight
    
    return score

# Ranking de modelos en test set
model_ranking = []
for model_name, results in models_results.items():
    test_metrics = results['metrics']['test']
    composite_score = calculate_composite_score(test_metrics)
    
    model_ranking.append({
        'Model': model_name,
        'Composite_Score': composite_score,
        'ROC_AUC': test_metrics['roc_auc'],
        'PR_AUC': test_metrics['pr_auc'],
        'F1': test_metrics['f1'],
        'Precision': test_metrics['precision'],
        'Recall': test_metrics['recall']
    })

ranking_df = pd.DataFrame(model_ranking).sort_values('Composite_Score', ascending=False)

print("\nRanking Final de Modelos (Test Set):")
print("-" * 80)
print(ranking_df.round(4))

# Mejor modelo
best_model_name = ranking_df.iloc[0]['Model']
best_model_info = models_results[best_model_name]

print(f"\n🏆 MEJOR MODELO: {best_model_name}")
print("-" * 50)
print(f"Score Compuesto: {ranking_df.iloc[0]['Composite_Score']:.4f}")
print(f"ROC-AUC: {ranking_df.iloc[0]['ROC_AUC']:.4f}")
print(f"PR-AUC: {ranking_df.iloc[0]['PR_AUC']:.4f}")
print(f"F1-Score: {ranking_df.iloc[0]['F1']:.4f}")

print(f"\nMejores Hiperparámetros:")
for param, value in best_model_info['best_params'].items():
    print(f"  {param}: {value}")

# Reporte de clasificación detallado del mejor modelo
print(f"\nReporte de Clasificación Detallado - {best_model_name}:")
print("-" * 60)
y_true_test, y_pred_test, _ = best_model_info['predictions']['test']
print(classification_report(y_true_test, y_pred_test, 
                          target_names=['No Attrition', 'Attrition']))

# Guardar resultados
results_summary = {
    'best_model': best_model_name,
    'ranking': ranking_df,
    'detailed_results': models_results
}

print(f"\n✅ ANÁLISIS COMPLETADO")
print(f"Total de modelos evaluados: {len(models_results)}")
print(f"Mejor modelo: {best_model_name}")
print(f"Score en test set: {ranking_df.iloc[0]['Composite_Score']:.4f}")

print("\n" + "="*80)
print("FIN DEL ANÁLISIS - TODOS LOS MODELOS ENTRENADOS Y EVALUADOS")
print("="*80)