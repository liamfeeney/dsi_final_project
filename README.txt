Phase 1
Outputs:
	Clean dataset - 
	Feature-engineered dataset - 
Helper Module
	phase1.py

Phase 2: 
Outputs:
	Key features - features.csv
	Input for ML phase - data_for_models.csv

Phase 3: 
Outputs:
	logistic regression model - logreg.pkl (includes list of used features)

Phase 4:
Outputs: 
	Naive Bayes - nb.pkl
	Decision Tree - clf.pkl
	Random Forest - rf.pkl
	SVM - svm.pkl (includes scaler)
	XGBoost - xgb.pkl
	
Helper Module:
	capstone.py
	
Phase 5: 
Outputs: 
	
Helper Module:
	capstone.py

Phase 6: 
Outputs:
	requirements.txt
	phaseVI.py
	
Helper Module:
	phase6.py
	phase1.py


To Deploy: 
1. Navigate to directory with requirements.txt, phaseIV.py, 
phase6.py, phase1.py, all the model .pkls and data_for_models.csv
2. Open a terminal window and run pip install -r requirements.txt
3. When that is finished run python phaseVI.py
4. Deployment should open automatically in local IP 127.000.00






