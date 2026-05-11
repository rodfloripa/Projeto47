

<p align="justify"><h1>Planejamento Fatorial em Machine Learning: Classificação da Qualidade de Vinhos</h1></p>

<p align="justify">Este projeto aplica técnicas de <b>Planejamento de Experimentos (DOE – Design of Experiments)</b> para analisar o impacto de diferentes escolhas de pré-processamento e algoritmos de Machine Learning na tarefa de classificação da qualidade de vinhos.</p>

<p align="justify">A proposta utiliza um <b>planejamento fatorial completo $2^4$</b>, permitindo avaliar sistematicamente como fatores como modelo, normalização, PCA e balanceamento afetam o desempenho do sistema.</p>

---

<p align="justify"><h2>Objetivo</h2></p>

<p align="justify">O objetivo principal do projeto é identificar quais componentes do pipeline de Machine Learning possuem maior influência sobre a métrica AUC (Area Under the ROC Curve), além de analisar possíveis interações entre os fatores experimentais.</p>

<p align="justify">Em vez de testar configurações aleatoriamente, o projeto utiliza metodologia estatística para construir uma análise estruturada e reprodutível.</p>

---

<p align="justify"><h2>Dataset</h2></p>

<p align="justify">Foi utilizada a base <b>Wine Quality Dataset</b>, disponibilizada pela UCI Machine Learning Repository.</p>

<p align="justify">A base contém atributos físico-químicos de vinhos tintos, incluindo:</p>

<p align="justify">
- Acidez fixa;<br>
- Acidez volátil;<br>
- Ácido cítrico;<br>
- Açúcar residual;<br>
- Cloretos;<br>
- Densidade;<br>
- pH;<br>
- Sulfatos;<br>
- Teor alcoólico.
</p>

<p align="justify">O problema original de regressão foi transformado em classificação binária:</p>
wine["quality_bin"] = (wine["quality"] >= 7).astype(int)
<p align="justify">
- Qualidade maior ou igual a 7 → vinho bom;<br>
- Qualidade menor que 7 → vinho ruim.
</p>

---

<p align="justify"><h2>Planejamento Fatorial</h2></p>

<p align="justify">O experimento utiliza um planejamento fatorial completo $2^4$, composto por quatro fatores binários.</p>
Fator	Descrição	-1	+1
A	Modelo	Logistic Regression	Random Forest
B	Scaling	Sem Scaling	StandardScaler
C	PCA	Sem PCA	PCA
D	Balanceamento	RandomUnderSampler	SMOTE
<p align="justify">Cada combinação experimental foi executada três vezes para reduzir variabilidade estatística.</p>

---

<p align="justify"><h2>Pipeline Experimental</h2></p>
Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

Balanceamento
if sampling == "smote":
    sampler = SMOTE(random_state=42)
else:
    sampler = RandomUnderSampler(random_state=42)

X_res, y_res = sampler.fit_resample(X_train, y_train)

Scaling opcional
if use_scaler:
    scaler = StandardScaler()
    X_res = scaler.fit_transform(X_res)
    X_test_proc = scaler.transform(X_test)
else:
    X_test_proc = X_test.copy()

PCA opcional
if use_pca:
    pca = PCA(n_components=0.95)
    X_res = pca.fit_transform(X_res)
    X_test_proc = pca.transform(X_test_proc)
---

<p align="justify"><h2>Modelos Avaliados</h2></p>

<p align="justify"><b>Logistic Regression</b></p>
log_reg = LogisticRegression(max_iter=1000)

param_grid_lr = {
    "C": [0.01, 0.1, 1, 10]
}
<p align="justify"><b>Random Forest</b></p>
rf = RandomForestClassifier(random_state=42)

param_grid_rf = {
    "n_estimators": [50, 100, 200]
}
<p align="justify">Os hiperparâmetros foram ajustados utilizando <b>GridSearchCV</b> com validação cruzada de 5 folds.</p>

---

<p align="justify"><h2>Análise Estatística</h2></p>

<p align="justify">Após a execução dos experimentos, foi realizada uma análise ANOVA para medir a influência estatística dos fatores.</p>
modelo = ols(
    'AUC ~ A + B + C + D + A:B + A:C + A:D + B:C + B:D + C:D',
    data=df_resultados
).fit()

anova = sm.stats.anova_lm(modelo, typ=2)
print(anova)
<p align="justify">Também foram gerados:</p>

<p align="justify">
- Gráfico de Pareto;<br>
- Gráficos de efeitos principais;<br>
- Comparações médias entre tratamentos;<br>
- Análise das interações entre fatores.
</p>

---

<p align="justify"><h2>Métricas Utilizadas</h2></p>

<p align="justify">A principal métrica utilizada foi:</p>
roc_auc_score(y_test, probs)
<p align="justify">Além disso, também foram analisadas:</p>

<p align="justify">
- Precision;<br>
- Recall;<br>
- F1-Score;<br>
- Matriz de confusão.
</p>

---

<p align="justify"><h2>Melhor Configuração</h2></p>
best_model.fit(X_res, y_res)

probs = best_model.predict_proba(X_test_proc)[:, 1]
preds = best_model.predict(X_test_proc)

auc = roc_auc_score(y_test, probs)

print("AUC:", auc)
print(classification_report(y_test, preds))
<p align="justify">A melhor configuração foi selecionada com base na maior média de AUC observada durante os experimentos.</p>

---

<p align="justify"><h2>Tecnologias Utilizadas</h2></p>
pandas
numpy
scikit-learn
imbalanced-learn
statsmodels
matplotlib
seaborn
---

<p align="justify"><h2>Estrutura do Projeto</h2></p>
.
├── Experimentos.ipynb
├── winequality-red.csv
├── requirements.txt
└── README.md
---

<p align="justify"><h2>Conclusão</h2></p>

<p align="justify">Este projeto demonstra como técnicas clássicas de Planejamento de Experimentos podem ser integradas ao Machine Learning para produzir análises mais robustas e interpretáveis.</p>

<p align="justify">Além de encontrar a melhor configuração de pipeline, o trabalho também permite compreender estatisticamente quais fatores realmente impactam o desempenho do modelo.</p>

<p align="justify">A utilização de DOE em Machine Learning é especialmente útil em cenários nos quais existem múltiplas combinações possíveis de pré-processamento, balanceamento e algoritmos, reduzindo experimentação aleatória e tornando o processo mais científico e reprodutível.</p>

Pronto, agora tá legível. 

Quer que eu também revise o texto pra deixar mais natural em português acadêmico, ou só precisava corrigir a codificação mesmo?
