/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2018 HPCC Systems®.  All rights reserved.
############################################################################## */
IMPORT $ AS ML_Core;
IMPORT ML_Core.Types;

DiscreteField := Types.DiscreteField;
NumericField := Types.NumericField;
t_Work_Item := Types.t_Work_Item;
t_RecordId := Types.t_RecordId;
t_FieldNumber := Types.t_FieldNumber;
t_FieldReal := Types.t_FieldReal;
t_Discrete := Types.t_Discrete;
Class_Stats := Types.Class_Stats;
Confusion_Detail := Types.Confusion_Detail;
Classification_Accuracy := Types.Classification_Accuracy; // Return structure for
                                                          // Classification.Accuracy
Class_Accuracy := Types.Class_Accuracy; // Return structure for Classification.AccuracyByClass
Regression_Accuracy := Types.Regression_Accuracy; // Return structure for Regression.Accuracy
AUC_Result := Types.AUC_Result; // Return structure for Classification.AUC_Score

/**
  * Analyze and assess the effectiveness of a Machine
  * Learning model.
  * <p>Sub-modules provide support for both Classification and Regression.
  *
  * <p>Each of the functions in this module support multi-work-item (i.e. Myriad interface) data, as well as
  * multi-variate data (supported by some ML bundles).  The number field, which is usually
  * = 1 for uni-variate data is used to distinguish multiple regressors in the case of multi-
  * variate models.
  *
  **/
EXPORT Analysis := MODULE
  /**
    * This sub-module provides functions for analyzing and assessing the effectiveness of
    * an ML Classification model.  It can be used with any ML Bundle that supports classification.
    */
  EXPORT Classification := MODULE
    /**
      * Given a set of expected dependent values, assess the number and percentage of records that
      * were of each class.
      *
      * @param actual The set of training-data or test-data dependent values in DATASET(DiscreteField)
      *               format.
      * @return DATASET(Class_Stats), one record per work-item, per classifier (i.e. number field) per
      *         class.
      * @see ML_Core.Types.Class_Stats
      **/
    EXPORT DATASET(Class_Stats) ClassStats(DATASET(DiscreteField) actual) := FUNCTION
      // Returns for each class: label, count, pct
      recStats := TABLE(actual, {wi, number, cnt := COUNT(GROUP)}, wi, number);
      cStats := TABLE(actual, {wi, number, value, cnt := COUNT(GROUP)}, wi, number, value);
      outStats := JOIN(cStats, recStats, LEFT.wi = RIGHT.wi AND LEFT.number = RIGHT.number,
                      TRANSFORM(Class_Stats,
                                  SELF.classifier := LEFT.number,
                                  SELF.class := LEFT.value,
                                  SELF.classCount := LEFT.cnt,
                                  SELF.classPct := LEFT.cnt / RIGHT.cnt,
                                  SELF := LEFT), LOOKUP);
      RETURN outStats;
    END; // ClassStats
    // Function to compare predicted and actual values and include them in a record set containing both,
    // as well as a 'correct' indicator, which is TRUE whenever the two match.
    SHARED CompareClasses(DATASET(DiscreteField) predicted, DATASET(DiscreteField) actual) := FUNCTION
      // Distribute predicted and actual by HASH32(wi, id)
      predD := DISTRIBUTE(predicted, HASH32(wi, id));
      actD := DISTRIBUTE(actual, HASH32(wi, id));
      cmp := JOIN(predD, actD, LEFT.wi = RIGHT.wi AND LEFT.number = RIGHT.number AND LEFT.id = RIGHT.id,
                  TRANSFORM({t_Work_Item wi, t_FieldNumber number, t_Discrete pred, t_discrete actual, BOOLEAN correct},
                              SELF.pred := LEFT.value,
                              SELF.actual := RIGHT.value,
                              SELF.correct := LEFT.value = RIGHT.value,
                              SELF := LEFT), LOCAL);
      // cmp is distributed by HASH32(wi, id)
      RETURN cmp;
    END; // CompareClasses
    /**
      * Returns the Confusion Matrix, counting the number of cases for each combination of predicted Class and
      * actual Class.
      *
      * @param predicted The predicted values for each id in DATASET(DiscreteField) format.
      * @param actual The actual (i.e. expected) values for each id in DATASET(DiscreteField) format.
      * @return DATASET(Confusion_Detail).  One record for each combination of work-item, number (i.e. classifier),
      *         predicted class, and actual class.
      * @see ML_Core.Types.Confusion_Detail
      *
      **/
    EXPORT DATASET(Confusion_Detail) ConfusionMatrix(DATASET(DiscreteField) predicted, DATASET(DiscreteField) actual) := FUNCTION
      cmp := CompareClasses(predicted, actual);
      // Count the number of samples that were actually of each class
      actualClassTots := TABLE(cmp, {wi, number, actual, tot := COUNT(GROUP)}, wi, number, actual);
      // Count the number of samples that were predicted for each class
      predClassTots := TABLE(cmp, {wi, number, pred, tot := COUNT(GROUP)}, wi, number, pred);
      // Count the number of samples for each combination of actual and predicted
      cm0 := TABLE(cmp, {wi, t_FieldNumber classifier := number,
                          t_Discrete actual_class := actual, t_Discrete predict_class := pred,
                          UNSIGNED4 occurs := COUNT(GROUP), BOOLEAN correct := pred = actual}, wi, number, actual, pred);
      // Now calculate the proportions (of both actual and predicted values for each combination)
      cm1 := JOIN(cm0, actualClassTots, LEFT.wi = RIGHT.wi AND LEFT.classifier = RIGHT.number
                      AND LEFT.actual_class = RIGHT.actual,
                    TRANSFORM({RECORDOF(LEFT), t_FieldReal pctActual},
                                SELF.pctActual := LEFT.occurs / RIGHT.tot,
                                SELF := LEFT), LOOKUP);
      cm2 := JOIN(cm1, predClassTots, LEFT.wi = RIGHT.wi AND LEFT.classifier = RIGHT.number
                      AND LEFT.predict_class = RIGHT.pred,
                    TRANSFORM({RECORDOF(LEFT), t_FieldReal pctPred},
                                SELF.pctPred := LEFT.occurs / RIGHT.tot,
                                SELF := LEFT), LOOKUP);
      cm := PROJECT(cm2, Confusion_Detail);
      RETURN cm;
    END; // ConfusionMatrix
    /**
      * Assess the overall accuracy of the classification predictions.
      *
      * <p>ML_Core.Types.Classification_Accuracy provides a detailed description of the return values. 
      *
      * @param predicted The predicted values for each id in DATASET(DiscreteField) format.
      * @param actual The actual (i.e. expected) values for each id in DATASET(DiscreteField) format.
      * @return DATASET(Classification_Accuracy).  One record for each combination of work-item, and
      *         number (i.e. classifier).
      * @see ML_Core.Types.Classification_Accuracy
      *
      **/
    EXPORT DATASET(Classification_Accuracy) Accuracy(DATASET(DiscreteField) predicted, DATASET(DiscreteField) actual) := FUNCTION
      // Returns Raw, PoD, PoDE
      cStats := ClassStats(actual);
      numClasses := TABLE(cStats, {wi, classifier, UNSIGNED4 num_classes := COUNT(GROUP)}, wi, classifier);
      mostCommon0 := SORT(cStats, wi, classifier, -classCount);
      mostCommon := DEDUP(mostCommon0, wi, classifier);
      globStats := JOIN(numClasses, mostCommon, LEFT.wi = RIGHT.wi AND LEFT.classifier = RIGHT.classifier,
                        TRANSFORM({numClasses, UNSIGNED4 highestCnt},
                                  SELF.highestCnt := RIGHT.classCount,
                                  SELF := LEFT));
      cmp := CompareClasses(predicted, actual);
      cmpStats := TABLE(cmp, {wi, number, UNSIGNED4 corrCnt := COUNT(GROUP, correct), UNSIGNED4 totCnt := COUNT(GROUP)}, wi, number);
      outStats := JOIN(cmpStats, globStats, LEFT.wi = RIGHT.wi AND LEFT.number = RIGHT.classifier,
                        TRANSFORM(Classification_Accuracy,
                                   SELF.classifier := LEFT.number,
                                   SELF.errCnt := LEFT.totCnt - LEFT.corrCnt,
                                   SELF.recCnt := LEFT.totCnt,
                                   SELF.Raw_accuracy := LEFT.corrCnt / LEFT.totCnt,
                                   SELF.PoD := (LEFT.corrCnt -  LEFT.totCnt / RIGHT.num_classes) /
                                                  (LEFT.totCnt - LEFT.totCnt / RIGHT.num_classes),
                                   SELF.PoDE := (LEFT.corrCnt - RIGHT.highestCnt) /
                                                  (LEFT.totCnt - RIGHT.highestCnt),
                                   SELF := LEFT), LOOKUP);
      RETURN outStats;
    END; // Accuracy
    /**
      * Provides per class accuracy / relevance statistics (e.g. Precision / Recall,
      * False-positive Rate).
      * 
      * <p>ML_Core.Types.Class_Accuracy provides a detailed description of the return values. 
      *
      * @param predicted The predicted values for each id in DATASET(DiscreteField) format.
      * @param actual The actual (i.e. expected) values for each id in DATASET(DiscreteField) format.
      * @return DATASET(Class_Accuracy).  One record for each combination of work-item, number (i.e. classifier),
      *         and class.
      * @see ML_Core.Types.Class_Accuracy
      *
      **/
    EXPORT DATASET(Class_Accuracy) AccuracyByClass(DATASET(DiscreteField) predicted, DATASET(DiscreteField) actual) := FUNCTION
      // Returns Precision, Recall, False Positive Rate(FPR)
      allClasses0 := SORT(actual, wi, number, value);
      allClasses := DEDUP(actual, wi, number, value);
      cmp := CompareClasses(predicted, actual);
      // For each class, replicate all of the items not of that class so that we can analyze that class
      // with respect to its non-members (i.e. negatives).
      allClassPts := JOIN(cmp, allClasses, LEFT.wi = RIGHT.wi AND LEFT.number = RIGHT.number,
                          TRANSFORM({cmp, UNSIGNED class},
                                      SELF.class := RIGHT.value,
                                      SELF := LEFT), MANY, LOOKUP);
      allClassSumm := TABLE(allClassPts, {wi, number, class,
                             UNSIGNED4 precDenom := COUNT(GROUP, pred = class),
                             UNSIGNED4 TP := COUNT(GROUP, pred = class AND actual = class),
                             UNSIGNED4 FP := COUNT(GROUP, pred = class AND actual != class),
                             UNSIGNED4 recallDenom := COUNT(GROUP, actual = class),
                             UNSIGNED4 TN := COUNT(GROUP, actual != class AND pred != class),
                             UNSIGNED4 FN := COUNT(GROUP, actual = class AND pred != class),
                             }, wi, number, class);
      cStats := PROJECT(allClassSumm, TRANSFORM(Class_Accuracy,
                                            SELF.classifier := LEFT.number,
                                            SELF.precision := LEFT.TP / (LEFT.TP + LEFT.FP),
                                            SELF.recall := LEFT.TP / (LEFT.TP + LEFT.FN),
                                            SELF.FPR := LEFT.FP / (LEFT.FP + LEFT.TN),
                                            SELF := LEFT));
      RETURN cStats;
    END; // AccuracyByClass
    /**
      * CorrectBinary
      *
      * Function to check whether the given data only contains binary classifiers
      * and if so, to change the labels so that the higher label becomes 1 and the
      * other becomes 0.
      *
      * @param field The dataset to be checked in DATASET(DiscreteField) format
      * @return DATASET(DiscreteField) The corrected dataset.
      *
      */
    SHARED DATASET(DiscreteField) CorrectBinary(DATASET(DiscreteField) field) := FUNCTION
      g1 := TABLE(field, {wi,number,value},wi,number,value);
      isBad := MAX(TABLE(g1,{wi,number,cnt:=COUNT(GROUP)},wi,number),cnt)>2;
      corrected := PROJECT(field, TRANSFORM(DiscreteField,
                                            SELF.value := IF(LEFT.value = MAX(g1(wi=LEFT.wi and number=LEFT.number),
                                                                              g1.value),1,0),
                                            SELF := LEFT));
      RETURN IF(isBad, FAIL(corrected, 'Multi-label data not supported'), corrected);
    END; // CorrectBinary
    /**
      * AUC_Score
      *
      * Provides the Area Under ROC Curve score. This score is calculated as the probability
      * that a random positive sample is ranked higher than a random negative sample.
      * This metric may only be used for the evaluation of binary classifiers, that provide
      * the confidence or the probability that a sample is positive / belongs to a class.
      *
      * @param pred The predicted results in DATASET(Classify_Result) format
      * @param actual The actual values in DATASET(DiscreteField) format
      * @return DATASET(AUC_Result) One record per work item, per classifier, containing the
      *                             the AUC score.
      * @see ML_Core.Types.AUC_Result
      *
      */
    EXPORT DATASET(AUC_Result) AUC_Score(DATASET(Classify_Result) pred, DATASET(DiscreteField) actual) := FUNCTION
      // Check and convert the given data into 0's and 1's
      Cactual := CorrectBinary(actual);
      Cpred := JOIN(CorrectBinary(pred), pred,
                    LEFT.wi=RIGHT.wi and
                    LEFT.id=RIGHT.id and
                    LEFT.number=RIGHT.number);
      // Combine predicted scores and actual classes
      comb := JOIN(Cpred, Cactual,
                   LEFT.wi=RIGHT.wi and
                   LEFT.id=RIGHT.id and
                   LEFT.number=RIGHT.number,
                   TRANSFORM({RECORDOF(Classify_Result), t_Discrete actual},
                             SELF.actual := RIGHT.value,
                             SELF.conf := (IF(LEFT.value=1,1,-1)*LEFT.conf/2) + 1,
                             SELF := LEFT));
      // All positive-negative pairs of points
      pairs := JOIN(comb, comb,
                    LEFT.wi = RIGHT.wi and
                    LEFT.number = RIGHT.number and
                    LEFT.actual = 1 and
                    RIGHT.actual = 0,
                    TRANSFORM({t_Work_Item wi, t_FieldNumber classifier, REAL isGood},
                              SELF.wi := LEFT.wi,
                              SELF.classifier := LEFT.number,
                              // IF scores are equal - 0.5
                              // IF scores match - 1
                              // IF scores don't match - 0
                              SELF.isGood := IF(LEFT.conf<>RIGHT.conf,IF(LEFT.conf>RIGHT.conf,1,0),0.5)));
      // Sum of values / Total number - gives AUC
      RETURN TABLE(pairs,{wi,classifier,AUC:=SUM(GROUP,isGood)/COUNT(GROUP)},wi,classifier);
    END; // AUC_Score
  END; // Classification
  /**
    * This sub-module provides functions for analyzing and assessing the effectiveness of
    * an ML Regression model.  It can be used with any ML Bundle that supports regression.
    *
    */
  EXPORT Regression := MODULE
    /**
      * Assess the overall accuracy of the regression predictions.
      *
      * <p>ML_Core.Types.Regression_Accuracy provides a detailed description of the return values. 
      *
      * @param predicted The predicted values for each id in DATASET(DiscreteField) format.
      * @param actual The actual (i.e. expected) values for each id in DATASET(DiscreteField) format.
      * @return DATASET(Regression_Accuracy).  One record for each combination of work-item, and
      *         number (i.e. regressor).
      * @see ML_Core.Types.Regression_Accuracy
      *
      **/
    EXPORT DATASET(Regression_Accuracy) Accuracy(DATASET(NumericField) predicted, DATASET(NumericField) actual) := FUNCTION
      // Returns R-squared, MSE, RMSE
      meanAct := TABLE(actual, {wi, number, REAL mean := AVE(GROUP, value), cnt := COUNT(GROUP)}, wi, number);
      cmp := JOIN(actual, predicted, LEFT.wi = RIGHT.wi AND LEFT.number = RIGHT.number AND LEFT.id = RIGHT.id,
                    TRANSFORM({t_Work_Item wi, t_FieldNumber number, t_FieldReal actual, t_FieldReal pred},
                      SELF.actual := LEFT.value,
                      SELF.pred := RIGHT.value,
                      SELF := LEFT));
      calc0 := JOIN(cmp, meanAct, LEFT.wi = RIGHT.wi AND LEFT.number = RIGHT.number,
                      TRANSFORM({cmp, REAL ts, REAL rs},
                        SELF.ts := POWER(LEFT.actual - RIGHT.mean, 2), // Total squared
                        SELF.rs := POWER(LEFT.actual - LEFT.pred, 2),  // Residual squared
                        SELF := LEFT), LOOKUP);
      // R2 := 1 - (Residual Sum of Squares / Total Sum of Squares)
      calc1 := TABLE(calc0, {wi, number, R2 := 1 - SUM(GROUP, rs) / SUM(GROUP, ts), RSS := SUM(GROUP, rs)}, wi, number);
      result := JOIN(calc1, meanAct, LEFT.wi = RIGHT.wi AND LEFT.number = RIGHT.number,
                      TRANSFORM(Regression_Accuracy,
                        SELF.MSE := LEFT.RSS / RIGHT.cnt,
                        SELF.RMSE := POWER(SELF.MSE, .5),
                        SELF.regressor := LEFT.number,
                        SELF := LEFT), LOOKUP);
      RETURN result;
    END; // Accuracy
  END; // Regression
END; // Analysis
