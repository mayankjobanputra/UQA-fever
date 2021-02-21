"""Generate accuracy scores and visualize them."""
from constants import RESULT_PATH, THRESHOLD
from dataset_reader import DataReader
import statistics


class Analyzer(object):
    """Analyze the predictions and generate the accuracy and plot them."""

    def __init__(self, r_path=RESULT_PATH):
        """Initialize Analyzer with the result data."""
        self.reader = DataReader(r_path)
        self.data = self.reader.get_data_from_json()

    def calculate_label_accuracy(self, tags=None):
        """Calculate label accuracy."""
        total_labels = len(self.data)
        correct_labels = 0
        tags = ["SUPPORTS"] if not tags else tags
        for obj in self.data:
            if obj['label'] in tags:
                predicted_label = self._get_predicted_label(obj)
                correct_labels += 1 if obj['label'] == predicted_label else 0
            else:
                total_labels -= 1
        return float(correct_labels)*100/float(total_labels)

    def get_incorrectly_tokenized(self):
        """Return Total Incorrectly tokenized questions."""
        supported_but_incorrectly_tokenized = incorrectly_tokenized = total = 0
        data = self.reader.get_data_from_json()
        for obj in data:
            for qas in obj['qas']:
                total += 1
                if not qas['tokenized_corectly']:
                    incorrectly_tokenized += 1
                    supported_but_incorrectly_tokenized += 1 if obj['label'] == "SUPPORTS" else 0
        print("supported_but_incorrectly_tokenized: {}, total_incorrect: {}".format(supported_but_incorrectly_tokenized, incorrectly_tokenized))
        print("total_qas: {}".format(total))

    def _get_predicted_label(self, obj, threshold=THRESHOLD):
        q_cnt, ans_c_cnt = 0, 0
        for qas in obj['qas']:
            q_cnt += 1.0
            ans_c_cnt += 1.0 if qas['is_correct'] else 0
        try:
            # predicted_label = "MANUAL_REVIEW" if ans_c_cnt/q_cnt < threshold else "SUPPORTS"
            if q_cnt > 3:
                predicted_label = "MANUAL_REVIEW" if ans_c_cnt/q_cnt < threshold else "SUPPORTS"
            else:
                predicted_label = "MANUAL_REVIEW" if ans_c_cnt < 1 else "SUPPORTS"
        except:
            predicted_label = "MANUAL_REVIEW"
        return predicted_label

    def calculate_precision_recall_f1_score(self, tags=None):
        """Calculate Precision Recall and F1 Score."""
        total_supported_labels = len(self.data)
        tp = fp = fn = tn = 0
        not_sup = pred_not_sup = 0
        tags = ["SUPPORTS"] if not tags else tags
        for obj in self.data:
            # if obj['label'] in tags:
            predicted_label = self._get_predicted_label(obj)
            tp += 1 if obj['label'] == "SUPPORTS" and predicted_label == obj['label'] else 0
            fp += 1 if predicted_label == "SUPPORTS" and obj['label'] != "SUPPORTS" else 0
            fn += 1 if obj['label'] == "SUPPORTS" and predicted_label != "SUPPORTS" else 0
            # tn += 1 if predicted_label != "SUPPORTS"  and obj['label'] != "SUPPORTS" else 0
            if obj['label'] != "SUPPORTS":
                not_sup += 1
            if predicted_label != "SUPPORTS":
                pred_not_sup += 1
            # else:
            #     total_supported_labels -= 1
        print("TP: {}, FP: {}, FN: {}, TN: {}".format(tp, fp, fn, tn))
        # print("total_supported_labels: {}, Not supp: {}, Pred Not Sup: {}".format(total_supported_labels, not_sup, pred_not_sup))
        precision = float(tp)*100/float(tp+fp)
        recall = float(tp)*100/float(tp+fn)
        return precision, recall, (2*precision*recall)/(precision+recall)

    def calculate_precision_recall_f1_score_thresh(self, tags=None, threshold=None):
        """Calculate Precision Recall and F1 Score."""
        total_supported_labels = len(self.data)
        not_sup = pred_not_sup = 0
        tags = ["SUPPORTS"] if not tags else tags
        threshold = [THRESHOLD] if not threshold else threshold
        precision_l, recall_l, f1_l = [], [], []
        for th in threshold:
            tp = fp = fn = tn = 0
            for obj in self.data:
                # if obj['label'] in tags:
                predicted_label = self._get_predicted_label(obj, th)
                tp += 1 if obj['label'] == "SUPPORTS" and predicted_label == obj['label'] else 0
                fp += 1 if predicted_label == "SUPPORTS" and obj['label'] != "SUPPORTS" else 0
                fn += 1 if obj['label'] == "SUPPORTS" and predicted_label != "SUPPORTS" else 0
                tn += 1 if predicted_label != "SUPPORTS"  and obj['label'] == predicted_label else 0
                if obj['label'] != "SUPPORTS":
                    not_sup += 1
                if predicted_label != "SUPPORTS":
                    pred_not_sup += 1
                # else:
                #     total_supported_labels -= 1
            print("Threshold: {}, TP: {}, FP: {}, FN: {}, TN: {}".format(th, tp, fp, fn, tn))
            # print("total_supported_labels: {}, Not supp: {}, Pred Not Sup: {}".format(total_supported_labels, not_sup, pred_not_sup))
            precision = float(tp)*100/float(tp+fp)
            recall = float(tp)*100/float(tp+fn)
            f1 = (2*precision*recall)/(precision+recall)
            precision_l.append(precision)
            recall_l.append(recall)
            f1_l.append(f1)
        return precision_l, recall_l, f1_l

    def calculate_avg_questions_per_cas(self):
        qas_len = 0
        total_ques = 0
        qas_stats = {}
        full_stats = []
        largest_claim = ""
        for obj in self.data:
            qas_len = len(obj['qas'])
            qas_stats[qas_len] = qas_stats[qas_len]+1 if qas_stats.get(qas_len) else 1
            total_ques += qas_len
            full_stats.append(qas_len)
            if qas_len == 13:
                largest_claim = obj
        print("Questions Frequency: {}".format(qas_stats))
        # print(total_ques)
        # print(largest_claim)
        print("Median (Questions per claim): {}".format(statistics.median(full_stats)))


if __name__ == "__main__":
    accuracy = Analyzer().calculate_label_accuracy()
    # Analyzer().get_incorrectly_tokenized()
    # precision, recall, f1 = Analyzer().calculate_precision_recall_f1_score()
    # print("Label accuracy on SUPPORTS label: {}, \nPrecision: {}, \nRecall: {}, \nf1_score: {}".format(accuracy, precision, recall, f1))
    th = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    precision_l, recall_l, f1_l = Analyzer().calculate_precision_recall_f1_score_thresh(threshold=th)
    print("Precision: {}, \n\nRecall: {}, \n\nf1_score: {}".format(precision_l, recall_l, f1_l))
    print()
    Analyzer().calculate_avg_questions_per_cas()
