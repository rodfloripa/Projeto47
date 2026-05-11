<p align="justify"><h1>Planejamento Fatorial em Machine Learning: ClassificaÃ§Ã£o da Qualidade de Vinhos</h1></p>

<p align="justify">Este projeto aplica tÃ©cnicas de <b>Planejamento de Experimentos (DOE â€” Design of Experiments)</b> para analisar o impacto de diferentes escolhas de prÃ©-processamento e algoritmos de Machine Learning na tarefa de classificaÃ§Ã£o da qualidade de vinhos.</p>

<p align="justify">A proposta utiliza um <b>planejamento fatorial completo \(2^4\)</b>, permitindo avaliar sistematicamente como fatores como modelo, normalizaÃ§Ã£o, PCA e balanceamento afetam o desempenho do sistema.</p>

---

<p align="justify"><h2>Objetivo</h2></p>

<p align="justify">O objetivo principal do projeto Ã© identificar quais componentes do pipeline de Machine Learning possuem maior influÃªncia sobre a mÃ©trica AUC (Area Under the ROC Curve), alÃ©m de analisar possÃ­veis interaÃ§Ãµes entre os fatores experimentais.</p>

<p align="justify">Em vez de testar configuraÃ§Ãµes aleatoriamente, o projeto utiliza metodologia estatÃ­stica para construir uma anÃ¡lise estruturada e reprodutÃ­vel.</p>

---

<p align="justify"><h2>Dataset</h2></p>

<p align="justify">Foi utilizada a base <b>Wine Quality Dataset</b>, disponibilizada pela UCI Machine Learning Repository.</p>

<p align="justify">A base contÃ©m atributos fÃ­sico-quÃ­micos de vinhos tintos, incluindo:</p>

<p align="justify">
- Acidez fixa;<br>
- Acidez volÃ¡til;<br>
- Ãcido cÃ­trico;<br>
- AÃ§Ãºcar residual;<br>
- Cloretos;<br>
- Densidade;<br>
- pH;<br>
- Sulfatos;<br>
- Teor alcoÃ³lico.
</p>

<p align="justify">O problema original de regressÃ£o foi transformado em classificaÃ§Ã£o binÃ¡ria:</p>

```python
wine["quality_bin"] = (wine["quality"] >= 7).astype(int)
```

<p align="justify">
- Qualidade maior ou igual a 7 â†’ vinho bom;<br>
- Qualidade menor que 7 â†’ vinho ruim.
</p>

---

<p align="justify"><h2>Planejamento Fatorial</h2></p>

<p align="justify">O experimento utiliza um planejamento fatorial completo \(2^4\), composto por quatro fatores binÃ¡rios.</p>

| Fator | DescriÃ§Ã£o | -1 | +1 |
|---|---|---|---|
| A | Modelo | Logistic Regression | Random Forest |
| B | Scaling | Sem Scaling | StandardScaler |
| C | PCA | Sem PCA | PCA |
| D | Balanceamento | RandomUnderSampler | SMOTE |

<p align="justify">Cada combinaÃ§Ã£o experimental foi executada trÃªs vezes para reduzir variabilidade estatÃ­stica.</p>

---

<p align="justify"><h2>Pipeline Experimental</h2></p>

```python
# Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Balanceamento
if sampling == "smote":
    sampler = SMOTE(random_state=42)
else:
    sampler = RandomUnderSampler(random_state=42)

X_res, y_res = sampler.fit_resample(X_train, y_train)

# Scaling opcional
if use_scaler:
    scaler = StandardScaler()
    X_res = scaler.fit_transform(X_res)
    X_test_proc = scaler.transform(X_test)
else:
    X_test_proc = X_test.copy()

# PCA opcional
if use_pca:
    pca = PCA(n_components=0.95)
    X_res = pca.fit_transform(X_res)
    X_test_proc = pca.transform(X_test_proc)
```

---

<p align="justify"><h2>Modelos Avaliados</h2></p>

<p align="justify"><b>Logistic Regression</b></p>

```python
log_reg = LogisticRegression(max_iter=1000)

param_grid_lr = {
    "C": [0.01, 0.1, 1, 10]
}
```

<p align="justify"><b>Random Forest</b></p>

```python
rf = RandomForestClassifier(random_state=42)

param_grid_rf = {
    "n_estimators": [50, 100, 200]
}
```

<p align="justify">Os hiperparÃ¢metros foram ajustados utilizando <b>GridSearchCV</b> com validaÃ§Ã£o cruzada de 5 folds.</p>

---

<p align="justify"><h2>AnÃ¡lise EstatÃ­stica</h2></p>

<p align="justify">ApÃ³s a execuÃ§Ã£o dos experimentos, foi realizada uma anÃ¡lise ANOVA para medir a influÃªncia estatÃ­stica dos fatores.</p>

```python
modelo = ols(
    'AUC ~ A + B + C + D + A:B + A:C + A:D + B:C + B:D + C:D',
    data=df_resultados
).fit()

anova = sm.stats.anova_lm(modelo, typ=2)
print(anova)
```

<p align="justify">TambÃ©m foram gerados:</p>

<p align="justify">
- GrÃ¡fico de Pareto;<br>
- GrÃ¡ficos de efeitos principais;<br>
- ComparaÃ§Ãµes mÃ©dias entre tratamentos;<br>
- AnÃ¡lise das interaÃ§Ãµes entre fatores.
</p>

---

<p align="justify"><h2>MÃ©tricas Utilizadas</h2></p>

<p align="justify">A principal mÃ©trica utilizada foi:</p>

```python
roc_auc_score(y_test, probs)
```

<p align="justify">AlÃ©m disso, tambÃ©m foram analisadas:</p>

<p align="justify">
- Precision;<br>
- Recall;<br>
- F1-Score;<br>
- Matriz de confusÃ£o.
</p>

---

<p align="justify"><h2>Melhor ConfiguraÃ§Ã£o</h2></p>

```python
best_model.fit(X_res, y_res)

probs = best_model.predict_proba(X_test_proc)[:, 1]
preds = best_model.predict(X_test_proc)

auc = roc_auc_score(y_test, probs)

print("AUC:", auc)
print(classification_report(y_test, preds))
```

<p align="justify">A melhor configuraÃ§Ã£o foi selecionada com base na maior mÃ©dia de AUC observada durante os experimentos.</p>

---

<p align="justify"><h2>Tecnologias Utilizadas</h2></p>

```python
pandas
numpy
scikit-learn
imbalanced-learn
statsmodels
matplotlib
seaborn
```

---

<p align="justify"><h2>Estrutura do Projeto</h2></p>

```python
.
â”œâ”€â”€ Experimentos.ipynb
â”œâ”€â”€ winequality-red.csv
â”œâ”€â”€ requirements.txt
â””â”€â”€ README.md
```

---

<p align="justify"><h2>ConclusÃ£o</h2></p>

<p align="justify">Este projeto demonstra como tÃ©cnicas clÃ¡ssicas de Planejamento de Experimentos podem ser integradas ao Machine Learning para produzir anÃ¡lises mais robustas e interpretÃ¡veis.</p>

<p align="justify">AlÃ©m de encontrar a melhor configuraÃ§Ã£o de pipeline, o trabalho tambÃ©m permite compreender estatisticamente quais fatores realmente impactam o desempenho do modelo.</p>

<p align="justify">A utilizaÃ§Ã£o de DOE em Machine Learning Ã© especialmente Ãºtil em cenÃ¡rios nos quais existem mÃºltiplas combinaÃ§Ãµes possÃ­veis de prÃ©-processamento, balanceamento e algoritmos, reduzindo experimentaÃ§Ã£o aleatÃ³ria e tornando o processo mais cientÃ­fico e reprodutÃ­vel.</p>
