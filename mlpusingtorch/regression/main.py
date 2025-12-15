import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt


# --- 1. CONFIGURAÇÃO ---
FILE_PATH = 'insurance.csv'
# Conteúdo CSV dummy para garantir que o código funcione se o arquivo não for encontrado
DUMMY_CSV_CONTENT = """age,sex,bmi,children,smoker,region,charges
19,female,27.9,0,yes,southwest,16884.924
18,male,33.77,1,no,southeast,1725.5523
28,male,33.0,3,no,southeast,4449.462
33,male,22.705,0,no,northwest,21984.47061
32,male,28.88,0,no,northwest,3866.8552
"""
NUM_EPOCHS = 1000
LEARNING_RATE = 0.001
BATCH_SIZE = 16

# Early Stopping Parameters
PATIENCE = 100
MIN_DELTA = 1e-1

# Configurar dispositivo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}")

# --- 2. PRÉ-PROCESSAMENTO E CARREGAMENTO DE DADOS ---

def load_and_preprocess(file_path):
    """Carrega CSV, codifica, escala e divide os dados."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Arquivo não encontrado em {file_path}. Usando dados dummy.")
        df = pd.read_csv(StringIO(DUMMY_CSV_CONTENT))

    # Engenharia de Features (One-Hot e Codificação Binária)
    df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True, dtype=int)
    
    # Separar Features (X) e Target (y)
    X = df.drop('charges', axis=1)
    y = df['charges'].values.astype(np.float32).reshape(-1, 1)

    # Dividir Dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Escalonamento (Standardization)
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = X_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train)

    X_test_scaled = X_scaler.transform(X_test)
    y_test_scaled = y_scaler.transform(y_test)
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler, y_test.flatten()

# Dataset Customizado PyTorch
class InsuranceDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = torch.from_numpy(X_data).float().to(DEVICE)
        self.y_data = torch.from_numpy(y_data).float().to(DEVICE)
        
    def __len__(self):
        return len(self.X_data)
    
    def __getitem__(self, idx):
        return self.X_data[idx], self.y_data[idx]

# Carregar e Preparar Data Loaders
X_train_s, X_test_s, y_train_s, y_test_s, y_scaler, y_test_original = load_and_preprocess(FILE_PATH)

train_dataset = InsuranceDataset(X_train_s, y_train_s)
test_dataset = InsuranceDataset(X_test_s, y_test_s)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

INPUT_DIM = X_train_s.shape[1]

# --- 3. DEFINIÇÃO DO MODELO ---

class InsurancePredictor(nn.Module):
    def __init__(self, input_size):
        super(InsurancePredictor, self).__init__()
        # Rede Neural Profunda (DNN) com 3 camadas ocultas e ReLU
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Camada de saída para regressão
        )

    def forward(self, x):
        return self.net(x)

# --- 4. TREINAMENTO COM EARLY STOPPING ---
model = InsurancePredictor(INPUT_DIM).to(DEVICE)
criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Early stopping setup
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None
train_losses = []
val_losses = []

print("\n--- Iniciando Treinamento com Early Stopping ---")
for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train() 
    epoch_train_loss = 0
    for X_batch, y_batch in train_loader:
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()      
        optimizer.step()
        epoch_train_loss += loss.item()
    
    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for X_val, y_val in test_loader:
            y_pred_val = model(X_val)
            val_loss += criterion(y_pred_val, y_val).item()
        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
    
    # Early stopping logic
    if avg_val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        print(f'Época [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} (New Best)')
    else:
        patience_counter += 1
        if (epoch + 1) % 100 == 0:
            print(f'Época [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} (Patience: {patience_counter}/{PATIENCE})')
    
    # Check for early stopping
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping triggered at epoch {epoch+1}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        break

# Load best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("Best model loaded for evaluation")

print("--- Treinamento Finalizado ---")

# --- 5. AVALIAÇÃO COM SEPARAÇÃO DE ERROS ---


def calculate_split_metrics(y_true, y_pred, scaler):
    """Calcula métricas, separando erros positivos e negativos."""
    
    # 1. Transformação Inversa para Dólares (Unidades Originais)
    y_pred_original = scaler.inverse_transform(y_pred.cpu().numpy().reshape(-1, 1)).flatten()
    
    # 2. Cálculo do Erro (Erro = Previsão - Real)
    errors = y_pred_original - y_true
    
    # --- Métricas de Erro Separadas ---
    
    # Erros Positivos (Superestimação): Previsão > Real (Risco de Perda Financeira para a seguradora)
    positive_errors = errors[errors > 0]
    
    # Erros Negativos (Subestimação): Previsão < Real (Risco de Insuficiência de Capital ou Sub-seguro)
    negative_errors = errors[errors < 0] * -1 # Multiplica por -1 para obter o valor absoluto do erro
    
    # --- Geração de Métricas ---
    
    metrics = {}
    
    # 5.1. Métricas de Superestimação (Previsão > Real)
    metrics['N_Over_Predictions'] = len(positive_errors)
    metrics['Mean_Over_Prediction_Error'] = np.mean(positive_errors) if len(positive_errors) > 0 else 0
    metrics['Std_Over_Prediction_Error'] = np.std(positive_errors) if len(positive_errors) > 1 else 0

    # 5.2. Métricas de Subestimação (Previsão < Real)
    metrics['N_Under_Predictions'] = len(negative_errors)
    metrics['Mean_Under_Prediction_Error'] = np.mean(negative_errors) if len(negative_errors) > 0 else 0
    metrics['Std_Under_Prediction_Error'] = np.std(negative_errors) if len(negative_errors) > 1 else 0
    
    # 5.3. Métricas Globais (Reutilizando as do código anterior)
    metrics['RMSE'] = np.sqrt(np.mean(errors**2))
    metrics['MAE'] = np.mean(np.abs(errors))
    metrics['R2'] = r2_score(y_true, y_pred_original)
    metrics['Mean_Error_Bias'] = np.mean(errors)




    # --- 3. CONFIGURAÇÃO E PLOTAGEM DO HISTOGRAMA ---

    # Criar a figura e os subplots (1 linha, 2 colunas)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Distribuição Separada dos Erros de Previsão em Dólares', fontsize=16)

    # --- Subplot 1: Superestimação (Risco de Perda) ---
    axes[0].hist(positive_errors, bins=15, color='red', alpha=0.7, edgecolor='black')

    # Adicionar a linha da Média do Erro Positivo
    if len(positive_errors) > 0:
        mean_pos_error = positive_errors.mean()
        axes[0].axvline(mean_pos_error, color='darkred', linestyle='dashed', linewidth=2, 
                        label=f'Média: ${mean_pos_error:,.0f}')
        
    axes[0].set_title('⬆️ Erros Positivos (Superestimação - Previsão > Real)')
    axes[0].set_xlabel('Magnitude do Erro (Dólares)')
    axes[0].set_ylabel('Frequência (Nº de Casos)')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.5)

    # --- Subplot 2: Subestimação (Risco de Sub-seguro) ---
    axes[1].hist(negative_errors, bins=15, color='green', alpha=0.7, edgecolor='black')

    # Adicionar a linha da Média do Erro Negativo
    if len(negative_errors) > 0:
        mean_neg_error = negative_errors.mean()
        axes[1].axvline(mean_neg_error, color='darkgreen', linestyle='dashed', linewidth=2, 
                        label=f'Média: ${mean_neg_error:,.0f}')

    axes[1].set_title('⬇️ Erros Negativos (Subestimação - Previsão < Real)')
    axes[1].set_xlabel('Magnitude do Erro (Dólares)')
    axes[1].set_ylabel('Frequência (Nº de Casos)')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.5)

    # Ajustar layout para evitar sobreposição e mostrar o gráfico
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show() # Para exibir o gráfico
    # plt.savefig('histograma_erros_separados_final.png') # Descomente para salvar o arquivo


    return metrics

# Colocar o modelo em modo de avaliação
model.eval()
with torch.no_grad():
    for X_test, _ in test_loader: # Não precisamos do y_test_scaled_tensor, já temos y_test_original
        y_test_pred_scaled = model(X_test)

# Calcular e exibir métricas
results = calculate_split_metrics(
    y_test_original, y_test_pred_scaled, y_scaler
)

# --- Exibição de Resultados ---

print("\n\n--- RELATÓRIO DE AVALIAÇÃO DE REGRESSÃO (Valores em Moeda) ---")
print("----------------------------------------------------------------")

print("💰 Métricas de Acurácia Global:")
print(f"  RMSE (Erro Quadrático Médio):      ${results['RMSE']:,.2f} (Erro típico em Dólares)")
print(f"  MAE (Erro Absoluto Médio):         ${results['MAE']:,.2f} (Erro médio simples)")
print(f"  R-squared (R²):                    {results['R2']:.4f} (Variância Explicada)")
print(f"  Viés Médio (Mean Error):           ${results['Mean_Error_Bias']:,.2f} (Tendência geral)")

print("\n⬆️ Análise de Superestimação (Previsão > Real) - Risco de Perda:")
print(f"  Nº de Casos:                       {results['N_Over_Predictions']}")
print(f"  Média do Erro Positivo:            ${results['Mean_Over_Prediction_Error']:,.2f}")
print(f"  Desvio Padrão do Erro Positivo:    ${results['Std_Over_Prediction_Error']:,.2f}")

print("\n⬇️ Análise de Subestimação (Previsão < Real) - Risco de Sub-seguro:")
print(f"  Nº de Casos:                       {results['N_Under_Predictions']}")
print(f"  Média do Erro Negativo:            ${results['Mean_Under_Prediction_Error']:,.2f}")
print(f"  Desvio Padrão do Erro Negativo:    ${results['Std_Under_Prediction_Error']:,.2f}")
print("----------------------------------------------------------------")

