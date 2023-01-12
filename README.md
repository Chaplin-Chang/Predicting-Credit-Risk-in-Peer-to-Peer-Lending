# Predicting Credit Risk in Peer-to-Peer Lending
Peer-to-peer (P2P) lending provides borrowers with relatively low borrowing interest rates and gives lenders a channel for investment on an online platform. Since most P2P lending does not require any guarantees, the overdue payment of borrowers results in a massive loss of lending platforms and lenders. Many risk prediction models are proposed to predict credit risk. However, these works build models with more than 50 features, which causes a lot of computation time. Besides, most P2P lending datasets are much more than the number of nondefault data. These researches ignore the data imbalance issue, leading to inaccurate predictions. Therefore, this study proposes a credit risk prediction system (CRPS) for P2P lending to solve data imbalance issues and only require few features to build the models. We implement a data preprocessing module, a feature selection module, a data synthesis module, and five risk prediction models in CRPS. In experiments, we evaluate CRPS based on the de-identified personal loan dataset of the LendingClub platform. The accuracy of the CRPS can achieve 99%, the recall reaches 0.95, and the F1-Score is 0.97. CRPS can accurately predict credit risk with less than 10 features and tackle data imbalance issues.

「P2P借貸之風險預測系統」透過分析借款人的各項借貸相關資料，評估借貸交易是否可能有逾期的情況發生，並在建置模型時解決資料不平衡的問題，提升模型準確度。本計畫完成一項提供給放款人P2P借貸信用違約的風險評估系統，違約預測準確率達99%，並將F1-Score提升至0.97，有效控制借貸違約率及降低借款人的違約風險、減少借貸平台因為違約而要向放款人賠償的成本支出，提供一個可信賴的風險控制指標。

## 使用說明
* 讀取訓練資料
   ```py
   X = pd.read_csv('x_train.csv')
   Y = pd.read_csv('y_train.csv')
  ```
* 填補空缺值
  * 零值：空缺值與其他現有數字無關。
  * 平均值：空缺值不得為0，如借款人申請之貸款金額。
  * 眾數：分類型特徵。
   ```py
   X.fillna(0,inplace=True)
   Y.fillna(0,inplace=True)
  ```
* 特徵選取
  * SFS：每次都選擇一個特徵加入，直到最優的特徵加入為止。
  * RFECV：重複排除不具影響力的特徵，對每次排出特徵後計算準確度，以準確度最高的特徵數目作為選定訓練特徵數目的依據。
  * Lasso：同時進行特徵選取和正則化的回歸分析方法，揭示了特徵的重要性。
   ```py
  min_features_to_select = 1  # Minimum number of features to consider
  rfecv = RFECV(estimator=XGBClassifier(), step=1, cv=StratifiedKFold(2),
              scoring=metrics.make_scorer(f1_score, average='weighted'),
              min_features_to_select=min_features_to_select)
  rfecv.fit(X, Y)

  print("Optimal number of features : %d" % rfecv.n_features_)
  print("Ranking %s" % rfecv.ranking_) # 重要程度排名
  print('Best features :', X.columns[rfecv.support_])
  
  # Plot number of features VS. cross-validation scores
  plt.figure()
  plt.xlabel("Number of features selected")
  plt.ylabel("Cross validation score (nb of correct classifications)")
  plt.plot(range(min_features_to_select,
               len(rfecv.grid_scores_) + min_features_to_select),
         rfecv.grid_scores_)
  plt.show()
  ```
* 資料轉換(解決資料不平衡問題)
  * SMOTE
  * Borderline SMOTE
   ```py
  from imblearn.over_sampling import BorderlineSMOTE
  #over_samples = SMOTE(random_state=1234) 
  over_samples_X,over_samples_y = BorderlineSMOTE(random_state=42, kind='borderline-1').fit_resample(X_train, y_train)
  #over_samples_X,over_samples_y = over_samples.fit_resample(X_train, y_train)
  ```
* 選擇預測模型
  * XGBoost
  * Random Forest
  * Logistic regression
  * CatBoost
  * Lasso Regression
  ```py
  # Splitting the trainset
  x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

  # Declare the model
  model = classifier

  # Cross_val_score(fbeta)
  scoring = cross_val_score(model, x_train, y_train, cv=5, 
                          scoring=metrics.make_scorer(f1_score, average='weighted'))
  print('Cross_val_score(mean):', scoring.mean())
  print('Corss_val_score(std):', scoring.std())

  # Fitting trainset
  model = model.fit(x_train, y_train)
  y_pred = model.predict(x_test)

  # Showing Score
  print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
  print('Recall:', metrics.recall_score(y_test, y_pred))
  print('fbeta:', metrics.fbeta_score(y_test, y_pred, beta=1.5))
  print('F1-score:', metrics.f1_score(y_test, y_pred))
   ```
* 讀取測試資料
   ```py
  test_X = pd.read_csv('x_train.csv')
  test_Y = pd.read_csv('y_train.csv')
  test_X.drop(['Data_id', 'id'], axis=1, inplace=True)
  test_Y.drop('Data_ID', axis=1, inplace=True)
  test_X = test_X[100000:]
  test_Y = test_Y[100000:]
  ```
* 測試 / 查看訓練結果
   ```py
  #Showing testing scores
  print('Accuracy:', metrics.accuracy_score(test_Y, pred))
  print('Recall:', metrics.recall_score(test_Y, pred))
  print('fbeta:', metrics.fbeta_score(test_Y, pred, beta=1.5))
  print('F1-score:', metrics.f1_score(test_Y, pred))
  print('Precision:', metrics.precision_score(test_Y, pred))
  ```
