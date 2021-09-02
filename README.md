# Predicting Credit Risk in Peer-to-Peer Lending
Peer-to-peer (P2P) lending provides borrowers with relatively low borrowing interest rates and gives lenders a channel for investment on an online platform. Since most P2P lending does not require any guarantees, the overdue payment of borrowers results in a massive loss of lending platforms and lenders. Many risk prediction models are proposed to predict credit risk. However, these works build models with more than 50 features, which causes a lot of computation time. Besides, most P2P lending datasets are much more than the number of nondefault data. These researches ignore the data imbalance issue, leading to inaccurate predictions. Therefore, this study proposes a credit risk prediction system (CRPS) for P2P lending to solve data imbalance issues and only require few features to build the models. We implement a data preprocessing module, a feature selection module, a data synthesis module, and five risk prediction models in CRPS. In experiments, we evaluate CRPS based on the de-identified personal loan dataset of the LendingClub platform. The accuracy of the CRPS can achieve 99%, the recall reaches 0.95, and the F1-Score is 0.97. CRPS can accurately predict credit risk with less than 10 features and tackle data imbalance issues.
