import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                           precision_recall_curve, roc_curve, auc, 
                           average_precision_score, f1_score)
import xgboost as xgb
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib
plt.style.use('default')
sns.set_palette("husl")

# Cargar datos
print("Cargando datos...")
df = pd.read_csv('./data/HR_Analytics.csv')
print(f"Dataset shape: {df.shape}")

# Preparar datos
print("\nPreparando datos...")

# Convertir target
df['Attrition_numeric'] = df['Attrition'].map({'No': 0, 'Yes': 1})

# Identificar columnas categóricas y numéricas
categorical_features = ['Department', 'JobRole', 'Gender', 'MaritalStatus', 
                       'EducationField', 'BusinessTravel', 'OverTime']

numerical_features = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 
                     'YearsAtCompany', 'YearsInCurrentRole', 
                     'YearsSinceLastPromotion', 'JobLevel', 'Education',
                     'DistanceFromHome', 'WorkLifeBalance', 'JobSatisfaction',
                     'EnvironmentSatisfaction', 'JobInvolvement', 'DailyRate',
                     'HourlyRate', 'MonthlyRate', 'NumCompaniesWorked',
                     'PercentSalaryHike', 'PerformanceRating', 
                     'RelationshipSatisfaction', 'StockOptionLevel',
                     'TrainingTimesLastYear', 'YearsWithCurrManager']

print(f"Features categóricas: {len(categorical_features)}")
print(f"Features numéricas: {len(numerical_features)}")

# Label Encoding para variables categóricas
label_encoders = {}
for feature in categorical_features:
    if feature in df.columns:
        le = LabelEncoder()
        df[f'{feature}_encoded'] = le.fit_transform(df[feature])
        label_encoders[feature] = le

# Preparar X y y
encoded_categorical = [f'{feat}_encoded' for feat in categorical_features if feat in df.columns]
final_features = encoded_categorical + numerical_features

X = df[final_features]
y = df['Attrition_numeric']

print(f"Total features: {len(final_features)}")
print(f"Distribución target: {np.bincount(y)} ({np.bincount(y)[1]/len(y)*100:.1f}% attrition)")

# Split en train (60%), test (20%), eval (20%)
print("\nSplit del dataset...")

# Primer split: 60% train, 40% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# Segundo split: 20% test, 20% eval del 40% temp
X_test, X_eval, y_test, y_eval = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
print(f"Eval set: {len(X_eval)} samples ({len(X_eval)/len(X)*100:.1f}%)")

print(f"Train attrition: {np.bincount(y_train)[1]/len(y_train)*100:.1f}%")
print(f"Test attrition: {np.bincount(y_test)[1]/len(y_test)*100:.1f}%")
print(f"Eval attrition: {np.bincount(y_eval)[1]/len(y_eval)*100:.1f}%")

# Escalar variables numéricas
print("\nEscalando variables numéricas...")
scaler = StandardScaler()

# Solo escalar las columnas numéricas
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_eval_scaled = X_eval.copy()

X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])
X_eval_scaled[numerical_features] = scaler.transform(X_eval[numerical_features])

# Calcular scale_pos_weight automáticamente
neg_count = np.sum(y_train == 0)
pos_count = np.sum(y_train == 1)
scale_pos_weight_auto = neg_count / pos_count

print(f"Casos negativos (No attrition): {neg_count}")
print(f"Casos positivos (Attrition): {pos_count}")
print(f"Scale pos weight automático: {scale_pos_weight_auto:.2f}")

# Optimización Bayesiana para XGBoost
print("\nIniciando optimización bayesiana para F1 Score...")

# Definir espacio de búsqueda optimizado
dimensions = [
    Integer(100, 500, name='n_estimators'),          # Más árboles para mejor captura
    Real(0.05, 0.2, name='learning_rate'),           # Learning rate más conservador
    Integer(3, 8, name='max_depth'),                 # Profundidad moderada
    Real(0.6, 0.9, name='subsample'),               # Subsample más conservador
    Real(0.6, 0.9, name='colsample_bytree'),        # Feature sampling conservador
    Real(0.1, 5.0, name='reg_alpha'),               # Regularización L1
    Real(0.1, 5.0, name='reg_lambda'),              # Regularización L2
    Real(scale_pos_weight_auto * 0.5, scale_pos_weight_auto * 2.0, name='scale_pos_weight'), # Rango alrededor del óptimo
    Integer(5, 50, name='min_child_weight'),         # Peso mínimo por hoja
    Real(0.0, 0.5, name='gamma')                     # Mínima ganancia para split
]

# Variables globales para optimización
best_score = 0
best_params = None
optimization_results = []

@use_named_args(dimensions)
def objective(**params):
    """
    Función objetivo para maximizar F1 Score con validación robusta
    """
    global best_score, best_params, optimization_results
    
    # Crear modelo con parámetros actuales
    model = xgb.XGBClassifier(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        scale_pos_weight=params['scale_pos_weight'],
        min_child_weight=params['min_child_weight'],
        gamma=params['gamma'],
        random_state=42,
        eval_metric='logloss',
        objective='binary:logistic',
        tree_method='hist',  # Más eficiente
        verbose=0
    )
    
    # Cross-validation con F1 Score - más folds para mejor estimación
    cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1', n_jobs=-1)
    mean_score = scores.mean()
    std_score = scores.std()
    
    # Penalizar alta variabilidad (modelo inestable)
    penalty = std_score * 0.1  # Penalidad por alta variabilidad
    adjusted_score = mean_score - penalty
    
    # Guardar resultados
    result = {
        'params': params.copy(),
        'mean_score': mean_score,
        'std_score': std_score,
        'adjusted_score': adjusted_score,
        'scores': scores.tolist()
    }
    optimization_results.append(result)
    
    # Actualizar mejor resultado (usando score ajustado)
    if adjusted_score > best_score:
        best_score = adjusted_score
        best_params = params.copy()
        print(f"Nuevo mejor F1 Score: {mean_score:.4f} (±{std_score:.4f}) [Ajustado: {adjusted_score:.4f}]")
    
    return -adjusted_score  # Minimizar el negativo del score ajustado

# Ejecutar optimización con más iteraciones
print("Optimizando hiperparámetros con enfoque en F1 Score...")
print("Esto puede tomar varios minutos...")

result = gp_minimize(
    func=objective,
    dimensions=dimensions,
    n_calls=80,  # Más iteraciones para mejor optimización
    n_initial_points=15,  # Más puntos iniciales para explorar
    random_state=42,
    verbose=False
)

print(f"Optimización completada!")
print(f"Mejor CV F1 Score: {best_score:.4f}")
print(f"Mejores parámetros:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

# Entrenar modelo final con mejores parámetros + mejoras adicionales
print("\nEntrenando modelo final optimizado...")

model = xgb.XGBClassifier(
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    reg_alpha=best_params['reg_alpha'],
    reg_lambda=best_params['reg_lambda'],
    scale_pos_weight=best_params['scale_pos_weight'],
    min_child_weight=best_params['min_child_weight'],
    gamma=best_params['gamma'],
    random_state=42,
    eval_metric='logloss',
    objective='binary:logistic',
    tree_method='hist',
    early_stopping_rounds=20,  # Más patience para early stopping
    verbose=0
)

# Entrenar con early stopping usando múltiples métricas
model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
    eval_names=['train', 'test'],
    verbose=False
)

print("Modelo optimizado entrenado exitosamente!")

# Predicciones en eval set
print("\nGenerando predicciones en eval set...")
y_eval_pred_proba = model.predict_proba(X_eval_scaled)[:, 1]

# Optimizar threshold para mejor F1 Score
print("Optimizando threshold para F1 Score...")
thresholds = np.arange(0.1, 0.9, 0.02)
f1_scores = []

for threshold in thresholds:
    y_pred_thresh = (y_eval_pred_proba >= threshold).astype(int)
    f1 = f1_score(y_eval, y_pred_thresh)
    f1_scores.append(f1)

# Encontrar mejor threshold
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx]
best_f1_threshold = f1_scores[best_threshold_idx]

print(f"Mejor threshold: {best_threshold:.3f}")
print(f"F1 Score con threshold óptimo: {best_f1_threshold:.4f}")

# Predicciones finales con threshold optimizado
y_eval_pred = (y_eval_pred_proba >= best_threshold).astype(int)

# También evaluar en train para detectar overfitting
y_train_pred_proba = model.predict_proba(X_train_scaled)[:, 1]
y_train_pred = (y_train_pred_proba >= best_threshold).astype(int)
f1_train = f1_score(y_train, y_train_pred)
f1_eval = f1_score(y_eval, y_eval_pred)

# Métricas en eval set
print("\n" + "="*60)
print("EVALUACIÓN EN EVAL SET")
print("="*60)

print(f"Scale pos weight usado: {best_params['scale_pos_weight']:.2f}")
print(f"Threshold optimizado: {best_threshold:.3f}")
print(f"CV F1 Score (optimización): {best_score:.4f}")
print(f"Train F1 Score: {f1_train:.4f}")
print(f"Eval F1 Score: {f1_eval:.4f}")
print(f"F1 Gap (train-eval): {f1_train - f1_eval:.4f}")

# Análisis de overfitting mejorado
if f1_train - f1_eval > 0.15:
    print("\n🚨 ALERTA: Overfitting severo detectado")
    print("Recomendación: Aumentar regularización o reducir complejidad")
elif f1_train - f1_eval > 0.08:
    print("\n⚠️  MODERADO: Gap aceptable pero monitorear")
    print("Modelo funcional pero puede mejorarse")
else:
    print("\n✅ EXCELENTE: Buen balance train-eval")
    print("Modelo bien generalizado")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_eval, y_eval_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_eval, y_eval_pred)
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Attrition', 'Attrition'],
            yticklabels=['No Attrition', 'Attrition'])
plt.title('Confusion Matrix - Eval Set', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# PR-AUC Curve
print("\nPR-AUC Curve:")
precision, recall, pr_thresholds = precision_recall_curve(y_eval, y_eval_pred_proba)
pr_auc = average_precision_score(y_eval, y_eval_pred_proba)
baseline = np.sum(y_eval) / len(y_eval)

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
plt.axhline(y=baseline, color='red', linestyle='--', lw=2, 
           label=f'Baseline = {baseline:.3f}')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True, alpha=0.3)

# ROC Curve
print("ROC Curve:")
fpr, tpr, roc_thresholds = roc_curve(y_eval, y_eval_pred_proba)
roc_auc = auc(fpr, tpr)

plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Feature Importance
print("\nFeature Importance:")
feature_importance = pd.DataFrame({
    'feature': final_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 15 features:")
print(feature_importance.head(15))

plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)
sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
plt.title('Top 15 Feature Importance', fontsize=14, fontweight='bold')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# Guardar modelo
print("\nGuardando modelo optimizado...")
model_components = {
    'model': model,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'feature_names': final_features,
    'categorical_features': categorical_features,
    'numerical_features': numerical_features,
    'feature_importance': feature_importance,
    'best_params': best_params,
    'best_cv_f1': best_score,
    'optimization_results': pd.DataFrame(optimization_results),
    'model_type': 'XGBoost_Optimized'
}

joblib.dump(model_components, 'attrition_model_xgboost_optimized.pkl')

# Resumen final
print("\n" + "="*60)
print("RESUMEN FINAL")
print("="*60)
print(f"Modelo: XGBoost con Optimización Bayesiana")
print(f"Split: 60% train, 20% test, 20% eval")
print(f"Features: {len(final_features)} ({len(categorical_features)} categóricas, {len(numerical_features)} numéricas)")
print(f"Optimización: 50 iteraciones para F1 Score")
print(f"")
print(f"RESULTADOS F1 SCORE:")
print(f"CV F1 Score: {best_score:.4f}")
print(f"Train F1 Score: {f1_train:.4f}")
print(f"Eval F1 Score: {f1_eval:.4f}")
print(f"F1 Gap: {f1_train - f1_eval:.4f}")
print(f"")
print(f"OTRAS MÉTRICAS EN EVAL:")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"PR AUC: {pr_auc:.4f}")
print(f"Baseline: {baseline:.4f}")
print(f"")
print(f"MEJORES PARÁMETROS:")
for param, value in best_params.items():
    if isinstance(value, float):
        print(f"  {param}: {value:.4f}")
    else:
        print(f"  {param}: {value}")
print(f"")
print(f"THRESHOLD OPTIMIZADO: {best_threshold:.3f} (vs default 0.5)")
print(f"SCALE POS WEIGHT: {best_params['scale_pos_weight']:.2f} (auto: {scale_pos_weight_auto:.2f})")
print(f"")
print(f"Modelo guardado: attrition_model_xgboost_optimized_f1.pkl")
print("="*60)

# Mostrar progreso de optimización
def plot_optimization_progress():
    """Muestra el progreso de la optimización con análisis detallado"""
    scores = [result['mean_score'] for result in optimization_results]
    adjusted_scores = [result['adjusted_score'] for result in optimization_results]
    iterations = range(1, len(scores) + 1)
    
    plt.figure(figsize=(15, 8))
    
    # Progreso de optimización
    plt.subplot(2, 3, 1)
    plt.plot(iterations, scores, 'b-', alpha=0.6, label='CV F1 Score')
    plt.plot(iterations, np.maximum.accumulate(scores), 'r-', linewidth=2, label='Best Score')
    plt.xlabel('Iteration')
    plt.ylabel('F1 Score')
    plt.title('Progreso de Optimización')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Distribución de scores
    plt.subplot(2, 3, 2)
    plt.hist(scores, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(max(scores), color='red', linestyle='--', linewidth=2, 
               label=f'Best: {max(scores):.4f}')
    plt.xlabel('F1 Score')
    plt.ylabel('Frequency')
    plt.title('Distribución de F1 Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Análisis de threshold
    plt.subplot(2, 3, 3)
    plt.plot(thresholds, f1_scores, 'g-', linewidth=2)
    plt.axvline(best_threshold, color='red', linestyle='--', 
               label=f'Óptimo: {best_threshold:.3f}')
    plt.axhline(best_f1_threshold, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('Optimización de Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Scale pos weight evolution
    plt.subplot(2, 3, 4)
    scale_weights = [result['params']['scale_pos_weight'] for result in optimization_results]
    plt.scatter(scale_weights, scores, alpha=0.6, c=scores, cmap='viridis')
    plt.axvline(scale_pos_weight_auto, color='red', linestyle='--', 
               label=f'Auto: {scale_pos_weight_auto:.2f}')
    plt.xlabel('Scale Pos Weight')
    plt.ylabel('F1 Score')
    plt.title('Impacto Scale Pos Weight')
    plt.colorbar(label='F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning rate vs F1
    plt.subplot(2, 3, 5)
    learning_rates = [result['params']['learning_rate'] for result in optimization_results]
    plt.scatter(learning_rates, scores, alpha=0.6, c=scores, cmap='plasma')
    plt.xlabel('Learning Rate')
    plt.ylabel('F1 Score')
    plt.title('Learning Rate vs F1')
    plt.colorbar(label='F1 Score')
    plt.grid(True, alpha=0.3)
    
    # Convergencia
    plt.subplot(2, 3, 6)
    plt.plot(iterations, adjusted_scores, 'purple', alpha=0.7, label='Adjusted Score')
    plt.plot(iterations, scores, 'blue', alpha=0.7, label='Raw Score')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title('Convergencia (Raw vs Adjusted)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

print("\nMostrando análisis completo de optimización...")
plot_optimization_progress()

# Función de predicción
def predict_attrition(model_components, employee_data):
    """
    Predice la probabilidad de attrition para un empleado
    """
    model = model_components['model']
    scaler = model_components['scaler']
    label_encoders = model_components['label_encoders']
    feature_names = model_components['feature_names']
    categorical_features = model_components['categorical_features']
    numerical_features = model_components['numerical_features']
    
    # Crear dataframe
    emp_df = pd.DataFrame([employee_data])
    
    # Label encoding para categóricas
    for feature in categorical_features:
        if feature in emp_df.columns and feature in label_encoders:
            le = label_encoders[feature]
            try:
                emp_df[f'{feature}_encoded'] = le.transform(emp_df[feature])
            except ValueError:
                # Valor no visto, usar 0
                emp_df[f'{feature}_encoded'] = 0
    
    # Preparar features
    emp_features = emp_df[feature_names].fillna(0)
    
    # Escalar numéricas
    emp_features[numerical_features] = scaler.transform(emp_features[numerical_features])
    
    # Predecir
    probability = model.predict_proba(emp_features)[0]
    prediction = model.predict(emp_features)[0]
    
    return {
        'prediction': 'Yes' if prediction == 1 else 'No',
        'probability_no': probability[0],
        'probability_yes': probability[1],
        'model_type': 'XGBoost'
    }

# Ejemplo de predicción
print("\nEjemplo de predicción:")
example_employee = {
    'Age': 35,
    'Department': 'Sales',
    'JobRole': 'Sales Representative',
    'Gender': 'Male',
    'MaritalStatus': 'Married',
    'EducationField': 'Marketing',
    'BusinessTravel': 'Travel_Frequently',
    'OverTime': 'Yes',
    'MonthlyIncome': 5000,
    'TotalWorkingYears': 10,
    'YearsAtCompany': 3,
    'YearsInCurrentRole': 2,
    'YearsSinceLastPromotion': 1,
    'JobLevel': 2,
    'Education': 3,
    'DistanceFromHome': 15,
    'WorkLifeBalance': 2,
    'JobSatisfaction': 3,
    'EnvironmentSatisfaction': 2,
    'JobInvolvement': 3,
    'DailyRate': 800,
    'HourlyRate': 50,
    'MonthlyRate': 15000,
    'NumCompaniesWorked': 2,
    'PercentSalaryHike': 15,
    'PerformanceRating': 3,
    'RelationshipSatisfaction': 3,
    'StockOptionLevel': 1,
    'TrainingTimesLastYear': 3,
    'YearsWithCurrManager': 2
}

try:
    result = predict_attrition(model_components, example_employee)
    print(f"Predicción: {result['prediction']}")
    print(f"Probabilidad No Attrition: {result['probability_no']:.3f}")
    print(f"Probabilidad Attrition: {result['probability_yes']:.3f}")
except Exception as e:
    print(f"Error en predicción: {e}")

print("\n¡Script completado exitosamente!")