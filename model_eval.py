import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
plt.style.use('default')
sns.set_palette("husl")

# Crear directorio de plots si no existe
PLOTS_DIR = './plots'
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)
    print(f"Directorio '{PLOTS_DIR}' creado.")

class TargetEncoder:
    """
    Target Encoder con suavizado para evitar overfitting
    """
    def __init__(self, smoothing=1.0, min_samples_leaf=1):
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.encodings = {}
        self.global_mean = None
        
    def fit(self, X, y):
        """
        Ajusta el encoder con los datos de entrenamiento
        """
        self.global_mean = y.mean()
        
        # Crear un DataFrame temporal combinando X e y
        temp_df = X.copy()
        temp_df['target'] = y.values
        
        for column in X.columns:
            # Calcular estadísticas por categoría
            stats = temp_df.groupby(column)['target'].agg(['count', 'mean']).reset_index()
            stats.columns = [column, 'count', 'mean']
            
            # Aplicar suavizado
            # Formula: (count * mean + smoothing * global_mean) / (count + smoothing)
            stats['smoothed_mean'] = (
                (stats['count'] * stats['mean'] + self.smoothing * self.global_mean) / 
                (stats['count'] + self.smoothing)
            )
            
            # Filtrar categorías con pocas muestras
            stats = stats[stats['count'] >= self.min_samples_leaf]
            
            # Guardar encoding
            encoding_dict = dict(zip(stats[column], stats['smoothed_mean']))
            self.encodings[column] = encoding_dict
            
        return self
    
    def transform(self, X):
        """
        Transforma las variables categóricas usando el encoding ajustado
        """
        X_encoded = X.copy()
        
        for column in X.columns:
            if column in self.encodings:
                # Mapear valores conocidos, usar global_mean para valores nuevos
                X_encoded[column] = X[column].map(self.encodings[column]).fillna(self.global_mean)
            
        return X_encoded
    
    def fit_transform(self, X, y):
        """
        Ajusta y transforma en un solo paso
        """
        return self.fit(X, y).transform(X)

class ModelAnalyzer:
    def __init__(self, model_path, data_path):
        """
        Inicializa el analizador del modelo
        
        Args:
            model_path: Ruta al archivo .pkl del modelo
            data_path: Ruta al archivo CSV con los datos
        """
        self.model_components = joblib.load(model_path)
        self.data = pd.read_csv(data_path)
        self.X_test = None
        self.y_test = None
        self.y_pred_proba = None
        self.feature_names = None
        
        # Preparar datos
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepara los datos de la misma manera que en el entrenamiento"""
        data = self.data.copy()
        
        # Convertir Attrition a numérico
        data['Attrition_numeric'] = data['Attrition'].map({'No': 0, 'Yes': 1})
        
        # Verificar si el modelo usa Target Encoding o Label Encoding
        model_type = self.model_components.get('model_type', 'RandomForest')
        
        if 'TargetEncoded' in model_type:
            # Nuevo modelo con Target Encoding
            target_encoder = self.model_components['target_encoder']
            scaler = self.model_components['scaler']
            categorical_features = self.model_components['categorical_features']
            numerical_features = self.model_components['numerical_features']
            feature_names = self.model_components['feature_names']
            
            # Preparar datos siguiendo el mismo proceso que en entrenamiento
            # Separar features categóricas y numéricas
            if categorical_features:
                X_categorical = data[categorical_features]
            else:
                X_categorical = pd.DataFrame()
                
            if numerical_features:
                X_numerical = data[numerical_features]
            else:
                X_numerical = pd.DataFrame()
            
            y = data['Attrition_numeric']
            
            # Dividir en train/test (mismo random_state que en entrenamiento)
            if len(categorical_features) > 0:
                X_cat_train, X_cat_test, y_train, y_test = train_test_split(
                    X_categorical, y, test_size=0.2, random_state=42, stratify=y
                )
                X_num_train, X_num_test = train_test_split(
                    X_numerical, test_size=0.2, random_state=42, stratify=y
                )[0:2]
            else:
                X_num_train, X_num_test, y_train, y_test = train_test_split(
                    X_numerical, y, test_size=0.2, random_state=42, stratify=y
                )
                X_cat_train = X_cat_test = pd.DataFrame()
            
            # Aplicar Target Encoding a las variables categóricas de test
            if len(categorical_features) > 0:
                X_cat_test_encoded = target_encoder.transform(X_cat_test)
            else:
                X_cat_test_encoded = pd.DataFrame()
            
            # Escalar variables numéricas de test
            if len(numerical_features) > 0:
                X_num_test_scaled = pd.DataFrame(
                    scaler.transform(X_num_test),
                    columns=numerical_features,
                    index=X_num_test.index
                )
            else:
                X_num_test_scaled = pd.DataFrame()
            
            # Combinar features
            X_test = pd.concat([X_cat_test_encoded, X_num_test_scaled], axis=1)
            
            # Asegurar orden correcto de columnas
            X_test = X_test[feature_names]
            
        else:
            # Modelo anterior con Label Encoding
            label_encoders = self.model_components['label_encoders']
            
            categorical_features = ['Department', 'JobRole', 'Gender', 'MaritalStatus', 
                                   'EducationField', 'BusinessTravel', 'OverTime']
            
            # Encodear variables categóricas
            for feature in categorical_features:
                if feature in data.columns and feature in label_encoders:
                    le = label_encoders[feature]
                    # Manejar valores no vistos durante el entrenamiento
                    data[f'{feature}_encoded'] = data[feature].map(
                        dict(zip(le.classes_, le.transform(le.classes_)))
                    ).fillna(0)
            
            # Usar los mismos features del modelo
            feature_names = self.model_components['feature_names']
            
            # Crear X y y
            X = data[feature_names]
            y = data['Attrition_numeric']
            
            # Dividir en train/test (mismo random_state que en entrenamiento)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Escalar usando el scaler del modelo
            scaler = self.model_components['scaler']
            X_test = scaler.transform(X_test)
        
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        
        # Obtener predicciones
        model = self.model_components['model']
        self.y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        print(f"Datos preparados: {len(X_test)} muestras de prueba")
        print(f"Distribución de clases en test: {np.bincount(y_test)}")
        print(f"Tipo de modelo: {model_type}")
    
    def plot_feature_importance(self, top_n=15):
        """
        Grafica la importancia de características del modelo
        """
        feature_importance = self.model_components['feature_importance'].head(top_n)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_importance, y='feature', x='importance', palette='viridis')
        plt.title(f'Top {top_n} Características Más Importantes', fontsize=16, fontweight='bold')
        plt.xlabel('Importancia', fontsize=12)
        plt.ylabel('Características', fontsize=12)
        plt.tight_layout()
        
        # Mostrar valores en las barras
        for i, v in enumerate(feature_importance['importance']):
            plt.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=10)
        
        # Guardar plot
        plt.savefig(f'{PLOTS_DIR}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_importance
    
    def calculate_shap_values(self, sample_size=100):
        """
        Calcula y visualiza los valores SHAP
        """
        print("Calculando valores SHAP...")
        
        try:
            # Tomar una muestra más pequeña para evitar errores
            if len(self.X_test) > sample_size:
                sample_indices = np.random.choice(len(self.X_test), sample_size, replace=False)
                if isinstance(self.X_test, pd.DataFrame):
                    X_sample = self.X_test.iloc[sample_indices].values
                    y_sample = self.y_test.iloc[sample_indices]
                else:
                    X_sample = self.X_test[sample_indices]
                    y_sample = self.y_test.iloc[sample_indices]
            else:
                if isinstance(self.X_test, pd.DataFrame):
                    X_sample = self.X_test.values
                else:
                    X_sample = self.X_test
                y_sample = self.y_test
            
            model = self.model_components['model']
            
            # Crear explicador SHAP
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # Si es clasificación binaria, tomar los valores para la clase positiva
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Clase 'Yes' (attrition)
            
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
            plt.title('SHAP Summary Plot - Impacto de Características en Predicción', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{PLOTS_DIR}/shap_summary.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Bar plot de importancia promedio
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, 
                             plot_type="bar", show=False)
            plt.title('SHAP Feature Importance - Valor Absoluto Promedio', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{PLOTS_DIR}/shap_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Waterfall plot para una muestra específica
            if len(X_sample) > 0:
                plt.figure(figsize=(12, 8))
                # Crear objeto Explanation para la primera muestra
                explanation = shap.Explanation(
                    values=shap_values[0], 
                    base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                    data=X_sample[0],
                    feature_names=self.feature_names
                )
                shap.waterfall_plot(explanation, show=False)
                plt.title('SHAP Waterfall Plot - Ejemplo Individual', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f'{PLOTS_DIR}/shap_waterfall.png', dpi=300, bbox_inches='tight')
                plt.show()
            
            return shap_values
            
        except Exception as e:
            print(f"Error en cálculo de SHAP: {e}")
            print("Continuando sin análisis SHAP...")
            return None
    
    def plot_roc_curve(self):
        """
        Calcula y grafica la curva ROC
        """
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=12)
        plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)', fontsize=12)
        plt.title('Curva ROC (Receiver Operating Characteristic)', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"ROC AUC Score: {roc_auc:.4f}")
        return roc_auc, fpr, tpr, thresholds
    
    def plot_precision_recall_curve(self):
        """
        Calcula y grafica la curva Precision-Recall
        """
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_pred_proba)
        pr_auc = average_precision_score(self.y_test, self.y_pred_proba)
        
        # Baseline (proporción de clase positiva)
        baseline = np.sum(self.y_test) / len(self.y_test)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR Curve (AUC = {pr_auc:.3f})')
        plt.axhline(y=baseline, color='red', linestyle='--', lw=2,
                   label=f'Baseline (Random) = {baseline:.3f}')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (Sensibilidad)', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Curva Precision-Recall', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"PR AUC Score: {pr_auc:.4f}")
        print(f"Baseline (Random): {baseline:.4f}")
        return pr_auc, precision, recall, thresholds
    
    def plot_threshold_analysis(self):
        """
        Analiza diferentes umbrales de clasificación
        """
        # Calcular métricas para diferentes umbrales
        thresholds = np.arange(0.1, 1.0, 0.05)
        metrics = []
        
        for threshold in thresholds:
            y_pred = (self.y_pred_proba >= threshold).astype(int)
            
            tp = np.sum((y_pred == 1) & (self.y_test == 1))
            fp = np.sum((y_pred == 1) & (self.y_test == 0))
            tn = np.sum((y_pred == 0) & (self.y_test == 0))
            fn = np.sum((y_pred == 0) & (self.y_test == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            metrics.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'specificity': specificity
            })
        
        metrics_df = pd.DataFrame(metrics)
        
        # Graficar
        plt.figure(figsize=(12, 8))
        plt.plot(metrics_df['threshold'], metrics_df['precision'], 'o-', label='Precision', linewidth=2)
        plt.plot(metrics_df['threshold'], metrics_df['recall'], 's-', label='Recall', linewidth=2)
        plt.plot(metrics_df['threshold'], metrics_df['f1'], '^-', label='F1-Score', linewidth=2)
        plt.plot(metrics_df['threshold'], metrics_df['specificity'], 'd-', label='Specificity', linewidth=2)
        
        plt.xlabel('Umbral de Clasificación', fontsize=12)
        plt.ylabel('Valor de Métrica', fontsize=12)
        plt.title('Análisis de Umbrales de Clasificación', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim([0.1, 0.95])
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Encontrar el mejor umbral por F1-Score
        best_threshold = metrics_df.loc[metrics_df['f1'].idxmax(), 'threshold']
        best_f1 = metrics_df['f1'].max()
        
        print(f"Mejor umbral (F1-Score): {best_threshold:.2f}")
        print(f"Mejor F1-Score: {best_f1:.4f}")
        
        return metrics_df
    
    def show_model_info(self):
        """
        Muestra información detallada del modelo cargado
        """
        print("\n" + "="*60)
        print("INFORMACIÓN DEL MODELO")
        print("="*60)
        
        model_type = self.model_components.get('model_type', 'Unknown')
        print(f"Tipo de modelo: {model_type}")
        
        if 'best_params' in self.model_components:
            print(f"\nMejores parámetros encontrados:")
            best_params = self.model_components['best_params']
            for param, value in best_params.items():
                print(f"  {param}: {value}")
        
        if 'best_score' in self.model_components:
            print(f"\nMejor CV Score: {self.model_components['best_score']:.4f}")
        
        if 'TargetEncoded' in model_type:
            print(f"\nFeatures categóricas: {len(self.model_components['categorical_features'])}")
            print(f"Features numéricas: {len(self.model_components['numerical_features'])}")
            
            if 'target_encoder' in self.model_components:
                te = self.model_components['target_encoder']
                print(f"Target Encoder smoothing: {te.smoothing}")
                print(f"Global mean: {te.global_mean:.4f}")
        
        print(f"\nTotal features: {len(self.feature_names)}")
        print("="*60)

    def generate_complete_report(self):
        """
        Genera un reporte completo con todas las métricas
        """
        print("="*60)
        print("REPORTE COMPLETO DE ANÁLISIS DEL MODELO")
        print("="*60)
        
        # 0. Información del modelo
        self.show_model_info()
        
        # 1. Feature Importance
        print("\n1. IMPORTANCIA DE CARACTERÍSTICAS")
        print("-" * 40)
        feature_importance = self.plot_feature_importance()
        
        # 2. SHAP Values
        print("\n2. ANÁLISIS SHAP")
        print("-" * 40)
        try:
            shap_values = self.calculate_shap_values()
        except Exception as e:
            print(f"SHAP falló: {e}")
            shap_values = None
        
        # 3. ROC Curve
        print("\n3. CURVA ROC")
        print("-" * 40)
        roc_auc, fpr, tpr, roc_thresholds = self.plot_roc_curve()
        
        # 4. Precision-Recall Curve
        print("\n4. CURVA PRECISION-RECALL")
        print("-" * 40)
        pr_auc, precision, recall, pr_thresholds = self.plot_precision_recall_curve()
        
        # 5. Threshold Analysis
        print("\n5. ANÁLISIS DE UMBRALES")
        print("-" * 40)
        metrics_df = self.plot_threshold_analysis()
        
        # Resumen final
        print("\n" + "="*60)
        print("RESUMEN DE MÉTRICAS")
        print("="*60)
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print(f"PR AUC Score: {pr_auc:.4f}")
        print(f"Mejor F1-Score: {metrics_df['f1'].max():.4f}")
        print(f"Mejor Umbral: {metrics_df.loc[metrics_df['f1'].idxmax(), 'threshold']:.2f}")
        
        # Interpretación
        print("\nINTERPRETACIÓN:")
        if roc_auc > 0.8:
            print("✅ Excelente capacidad de discriminación (ROC AUC > 0.8)")
        elif roc_auc > 0.7:
            print("✓ Buena capacidad de discriminación (ROC AUC > 0.7)")
        else:
            print("⚠️ Capacidad de discriminación moderada (ROC AUC < 0.7)")
        
        if pr_auc > 0.5:
            print("✅ Buen rendimiento en clase minoritaria (PR AUC > 0.5)")
        else:
            print("⚠️ Rendimiento limitado en clase minoritaria (PR AUC < 0.5)")
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'feature_importance': feature_importance,
            'shap_values': shap_values,
            'metrics_by_threshold': metrics_df
        }

# Función principal para ejecutar el análisis
def main():
    """
    Función principal para ejecutar el análisis completo
    """
    # Configurar rutas
    MODEL_PATH = 'attrition_model_lightgbm_target_encoded.pkl'
    DATA_PATH = './data/HR_Analytics.csv'
    
    try:
        # Crear analizador
        analyzer = ModelAnalyzer(MODEL_PATH, DATA_PATH)
        
        # Ejecutar análisis completo
        results = analyzer.generate_complete_report()
        
        return analyzer, results
        
    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo {e}")
        print("Asegúrate de que las rutas del modelo y datos sean correctas")
        return None, None
    except Exception as e:
        print(f"Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    # Ejecutar análisis
    analyzer, results = main()
    
    # Solo procesar si no hubo errores
    if analyzer is not None and results is not None:
        print(f"\n¡Análisis completado exitosamente!")
        print(f"Plots guardados en: {PLOTS_DIR}/")
        print(f"- feature_importance.png")
        print(f"- roc_curve.png") 
        print(f"- precision_recall_curve.png")
        print(f"- threshold_analysis.png")
        if results.get('shap_values') is not None:
            print(f"- shap_summary.png")
            print(f"- shap_importance.png")
            print(f"- shap_waterfall.png")
    else:
        print("\nEl análisis no se pudo completar debido a errores.")