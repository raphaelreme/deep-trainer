import math

import pytest
import torch

from deep_trainer import metric as dt_metric


def test_pytorch_metric_cycle():
    def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - target).abs().mean()

    # Init
    m = dt_metric.PytorchMetric(mae)
    assert math.isnan(m.last_value)
    assert m._sum == 0
    assert m._n_samples == 0

    # Update
    # batch_size: 2, error 1.0
    outputs = torch.zeros((2, 1))
    targets = torch.full((2, 1), 1.0)
    m.update((None, targets), outputs)
    assert m.last_value == pytest.approx(1.0)
    assert m._sum == pytest.approx(2.0)

    # Aggregate
    assert m.aggregate() == pytest.approx(1.0)

    # Reset
    m.reset()
    assert math.isnan(m.last_value)
    assert m._sum == 0
    assert m._n_samples == 0


def test_pytorch_metric_weighted_average_by_batch_size():
    # loss_function returns per-batch mean absolute error
    def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - target).abs().mean()

    m = dt_metric.PytorchMetric(mae, display_name="MAE")

    # batch_size: 2, error 1.0
    outputs = torch.zeros((2, 1))
    targets = torch.full((2, 1), 1.0)
    m.update((None, targets), outputs)
    assert m.last_value == pytest.approx(1.0)

    # batch_size: 4, error 2.0
    outputs2 = torch.zeros((4, 1))
    targets2 = torch.full((4, 1), 2.0)
    m.update((None, targets2), outputs2)
    assert m.last_value == pytest.approx(2.0)

    # Weighted average by samples: (1*2 + 2*4) / 6 = 10/6
    assert m.aggregate() == pytest.approx(10.0 / 6.0)

    m.reset()
    assert math.isnan(m.last_value)
    assert m._sum == 0
    assert m._n_samples == 0


def test_accuracy_error_topk():
    # 3 classes, 4 samples
    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0],  # pred 0
            [0.0, 10.0, 0.0],  # pred 1
            [0.0, 0.0, 10.0],  # pred 2
            [0.0, 9.0, 10.0],  # pred 2, top2 contains 1 and 2
        ]
    )
    targets = torch.tensor([0, 1, 2, 1])

    acc = dt_metric.Accuracy()
    err = dt_metric.Error()
    top2 = dt_metric.TopK(2)

    acc.update((None, targets), logits)
    err.update((None, targets), logits)
    top2.update((None, targets), logits)

    assert acc.last_value == pytest.approx(3 / 4)
    assert err.last_value == pytest.approx(1 / 4)
    # last sample target is 1, top2 predictions are [2,1] => hit
    assert top2.last_value == pytest.approx(1.0)


def test_balanced_accuracy():
    logits = torch.tensor(
        [
            [10.0, 0.0],  # pred 0
            [10.0, 0.0],  # pred 0
            [0.0, 10.0],  # pred 1
            [10.0, 0.0],  # pred 0 (wrong if target=1)
        ]
    )
    targets = torch.tensor([0, 0, 1, 1])

    m = dt_metric.BalancedAccuracy()
    m.update((None, targets), logits)

    # Not defined per batch
    assert math.isnan(m.last_value)

    # class 0 recall: 2/2 = 1
    # class 1 recall: 1/2 = 0.5
    assert m.aggregate() == pytest.approx((1.0 + 0.5) / 2.0)

    m.reset()
    assert len(m.target_occurrences) == 0
    assert len(m.true_positives) == 0


def test_confusion_matrix_counts():
    cm = dt_metric.ConfusionMatrix(k=3)

    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0],  # pred 0
            [0.0, 10.0, 0.0],  # pred 1
            [0.0, 0.0, 10.0],  # pred 2
            [0.0, 10.0, 9.0],  # pred 1
        ]
    )
    targets = torch.tensor([0, 2, 2, 0])

    # Update
    cm.update((None, targets), logits)

    # true 0 predicted 0: 1
    # true 0 predicted 1: 1
    # true 2 predicted 1: 1
    # true 2 predicted 2: 1
    expected = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    assert torch.allclose(cm.confusion_matrix, expected)

    # Aggregate is no-op
    cm.aggregate()
    assert torch.allclose(cm.confusion_matrix, expected)

    # Reset
    cm.reset()
    assert torch.allclose(cm.confusion_matrix, torch.zeros(3, 3))


@pytest.mark.parametrize("average", ["macro", "weighted"])
def test_recall_precision_f1_from_confusion_matrix(average):
    # Confusion matrix for 3 classes:
    confusion = torch.tensor(
        [
            [5.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 0.0, 2.0],
        ]
    )
    # Compute expected per-class
    # occurrences (row sums): [6, 5, 2]
    # pred occurrences (col sums): [6, 4, 3]
    tp = torch.tensor([5.0, 3.0, 2.0])
    occ = torch.tensor([6.0, 5.0, 2.0])
    pred_occ = torch.tensor([6.0, 4.0, 3.0])

    recall_c = tp / occ
    prec_c = tp / pred_occ
    f1_c = 2.0 / (1.0 / prec_c + 1.0 / recall_c)

    if average == "macro":
        exp_recall = recall_c.mean().item()
        exp_prec = prec_c.mean().item()
        exp_f1 = f1_c.mean().item()
    else:
        weights = occ / occ.sum()
        exp_recall = (recall_c * weights).sum().item()  # equals tp.sum()/occ.sum()
        exp_prec = (prec_c * weights).sum().item()
        exp_f1 = (f1_c * weights).sum().item()

    cm = dt_metric.ConfusionMatrix(k=3)
    cm.confusion_matrix = confusion
    recall = dt_metric.Recall(cm, average=average)
    prec = dt_metric.Precision(cm, average=average)
    f1 = dt_metric.F1(cm, average=average)

    assert recall.aggregate() == pytest.approx(exp_recall)
    assert prec.aggregate() == pytest.approx(exp_prec)
    assert f1.aggregate() == pytest.approx(exp_f1)


def test_f1_raises_if_predicts_class_with_no_true_samples():
    # class 2 has no true samples (row 2 sum = 0) but is predicted (col 2 sum > 0)
    confusion = torch.tensor(
        [
            [3.0, 0.0, 1.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    cm = dt_metric.ConfusionMatrix(k=3)
    cm.confusion_matrix = confusion
    f1 = dt_metric.F1(cm, average="macro")

    with pytest.raises(ValueError, match="Cannot compute recall"):
        _ = f1.aggregate()


def test_recall_precision_f1_raises_if_wrong_average():
    cm = dt_metric.ConfusionMatrix(k=3)

    with pytest.raises(ValueError, match="`average` should be"):
        _ = dt_metric.Recall(cm, average="micro")

    with pytest.raises(ValueError, match="`average` should be"):
        _ = dt_metric.Precision(cm, average="micro")

    with pytest.raises(ValueError, match="`average` should be"):
        _ = dt_metric.F1(cm, average="micro")


def test_confusion_raises_with_shape_mismatch():
    cm = dt_metric.ConfusionMatrix(k=2)

    logits = torch.tensor(
        [
            [10.0, 0.0],  # pred 0
            [0.0, 10.0],  # pred 1
        ]
    )
    targets = torch.tensor([0, 1, 1])

    with pytest.raises(ValueError, match="Wrong predicted targets shape"):
        cm.update((None, targets), logits)


def test_balanced_accuracy_raises_with_shape_mismatch():
    m = dt_metric.BalancedAccuracy()

    logits = torch.tensor(
        [
            [10.0, 0.0],  # pred 0
            [0.0, 10.0],  # pred 1
        ]
    )
    targets = torch.tensor([0, 1, 1])

    with pytest.raises(ValueError, match="Wrong predicted targets shape"):
        m.update((None, targets), logits)


def test_metric_handler_sorting():
    update = []
    aggregate = []

    class A(dt_metric.Prerequisite):
        def update(self, _batch, _outputs):
            update.append(self.__class__.__name__)

        def aggregate(self):
            aggregate.append(self.__class__.__name__)

    class B(A):
        def __init__(self, prereq: A):
            super().__init__()
            self.prerequisites.add(prereq)

        def update(self, batch, outputs):
            return super().update(batch, outputs)

    class C(A):
        def __init__(self, prereq: A):
            super().__init__()
            self.prerequisites.add(prereq)

    class D(A):
        def __init__(self, prereq_b: B, prereq_c: C):
            super().__init__()
            self.prerequisites.add(prereq_b)
            self.prerequisites.add(prereq_c)

    class Metric(dt_metric.Metric):
        def update(self, _batch, _outputs):
            update.append("M")

        def aggregate(self):
            aggregate.append("M")
            return 0.0

    prereq = A()
    m = Metric()
    m.prerequisites.add(D(B(prereq), C(prereq)))

    handler = dt_metric.MetricsHandler([m])

    handler.update(None, None)

    assert update in (["A", "B", "C", "D", "M"], ["A", "C", "B", "D", "M"])

    metrics = handler.aggregated_values

    assert aggregate in (["A", "B", "C", "D", "M"], ["A", "C", "B", "D", "M"])
    assert metrics["Metric"] == 0.0


def test_set_validation_metric_raise_if_not_eval_metric():
    handler = dt_metric.MetricsHandler([dt_metric.Accuracy(evaluate=False)])

    with pytest.raises(ValueError, match="is not in evaluate mode"):
        handler.set_validation_metric(0)


def test_old_module_is_depreciated():
    with pytest.warns(DeprecationWarning, match="deep_trainer.pytorch.metric is deprecated"):
        import deep_trainer.pytorch.metric  # noqa: F401, PLC0415
