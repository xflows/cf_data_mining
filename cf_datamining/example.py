import cf_datamining.library as dm
import numpy as np

def myex1():

	# =======================================================
	# data	
	dsNum = dm.scikitAlgorithms_UCIDataset({'dsIn':'boston'})
	dsNom = dm.scikitAlgorithms_UCIDataset({'dsIn':'iris'})
	inst=dsNum['dtsOut']

	#====================================================

	par={'featureIn':'auto', 'depthIn':100}
	t = dm.scikitAlgorithms_J48(par)
	t = t['treeOut']

	# par={'learner':t, 'instances':inst}
	# res = dm.scikitAlgorithms_buildClassifier(par)

	par={'featureIn':'auto', 'depthIn':100}
	rt = dm.scikitAlgorithms_DecisionTreeRegressor(par)
	rt = rt['treeOut']

	par={'learner':rt, 'instances':inst}
	builtClassifier = dm.scikitAlgorithms_buildClassifier(par)


	# ===========================================
	# Apply-classifier

	par = {'classifier':builtClassifier['classifier'], 'data': inst}
	resAppl = dm.scikitAlgorithms_applyClassifier(par)
	# return resAppl['classes']

	res = resAppl['classes']

	resMSE = dm.scikitAlgorithms_MSE( {'data':res})
	print resMSE
	izl77


	import sklearn.metrics as met #import roc_auc_score
	resAcc = met.accuracy_score( res.target, res.targetPredicted)

	return resAcc
	
	# print res['classes'].keys(); izl77
	# print res.keys(); izl77
	# print res; izl8


# ===========================================
# Cross-validation

# X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
# X_train.shape, y_train.shape
# ((90, 4), (90,))
# X_test.shape, y_test.shape
# ((60, 4), (60,))
# clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
# >>> clf.score(X_test, y_test)



# ===========================================
# AUC
# from sklearn.metrics import roc_auc_score
# # metrics.roc_auc_score

# y_true = np.array([0, 0, 1, 1])
# y_scores = np.array([0.1, 0.4, 0.35, 0.8])
# res = roc_auc_score(y_true, y_scores)
# print str(res)
