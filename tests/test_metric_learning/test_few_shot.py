"""Tests for FewShotWrapper — N-way K-shot episodes."""

from collections import Counter

import pytest
from torch.utils.data import DataLoader

from torchdatasets.metric_learning import FewShotWrapper


@pytest.fixture
def five_classes(toy_dataset):
    """5 classes x 10 samples; sample i has class i // 10."""
    return toy_dataset([i // 10 for i in range(50)])


def ids_and_labels(episode_set):
    return [(int(sample.item()), label) for sample, label in episode_set]


class TestEpisodes:
    def test_sizes_and_episode_labels(self, five_classes):
        support, query = FewShotWrapper(five_classes, n_way=3, k_shot=2, q_query=4)[0]
        assert len(support) == 6 and len(query) == 12
        assert Counter(label for _, label in support) == {0: 2, 1: 2, 2: 2}
        assert Counter(label for _, label in query) == {0: 4, 1: 4, 2: 4}

    def test_each_episode_label_is_one_real_class_and_samples_differ(self, five_classes):
        wrapper = FewShotWrapper(five_classes, n_way=4, k_shot=3, q_query=5)
        for _ in range(20):
            support, query = wrapper[0]
            pairs = ids_and_labels(support) + ids_and_labels(query)
            assert len({sample for sample, _ in pairs}) == len(pairs)  # no sample used twice
            classes = {}
            for sample, label in pairs:
                classes.setdefault(label, set()).add(five_classes.targets[sample])
            assert all(len(real) == 1 for real in classes.values())
            assert len({next(iter(real)) for real in classes.values()}) == 4  # 4 different real classes

    def test_relabel_false_keeps_dataset_classes(self, five_classes):
        support, query = FewShotWrapper(five_classes, n_way=2, k_shot=1, q_query=1, relabel=False)[0]
        for sample, label in ids_and_labels(support) + ids_and_labels(query):
            assert label == five_classes.targets[sample]

    def test_transforms(self, five_classes):
        wrapper = FewShotWrapper(five_classes, n_way=2, k_shot=1, q_query=1,
                                 transform_support=lambda x: x + 100, transform_query=lambda x: x + 1000)
        support, query = wrapper[0]
        assert all(100 <= s.item() < 1000 for s, _ in support)
        assert all(s.item() >= 1000 for s, _ in query)


class TestLengthAndRandomness:
    def test_length_defaults_to_dataset_size(self, five_classes):
        assert len(FewShotWrapper(five_classes, k_shot=1, q_query=5)) == 50

    def test_default_sizes_need_16_samples_per_class(self, five_classes):
        with pytest.raises(ValueError, match="at least 16 samples"):
            FewShotWrapper(five_classes)  # 5-way 1-shot 15-query, but only 10 samples per class

    def test_n_episodes_sets_length_and_iteration_stops(self, five_classes):
        wrapper = FewShotWrapper(five_classes, q_query=5, n_episodes=3)
        assert len(wrapper) == 3 and len(list(wrapper)) == 3
        with pytest.raises(IndexError):
            wrapper[3]

    def test_random_state_fixes_each_episode(self, five_classes):
        a = FewShotWrapper(five_classes, n_way=3, k_shot=1, q_query=2, n_episodes=5, random_state=0)
        b = FewShotWrapper(five_classes, n_way=3, k_shot=1, q_query=2, n_episodes=5, random_state=0)
        episodes_a = [ids_and_labels(a[i][0]) + ids_and_labels(a[i][1]) for i in range(5)]
        assert episodes_a == [ids_and_labels(b[i][0]) + ids_and_labels(b[i][1]) for i in range(5)]
        assert len({tuple(e) for e in episodes_a}) > 1  # different indices, different episodes

    def test_dataloader_without_batching_yields_episodes(self, five_classes):
        loader = DataLoader(FewShotWrapper(five_classes, n_way=2, k_shot=1, q_query=1, n_episodes=2), batch_size=None)
        episodes = list(loader)
        assert len(episodes) == 2 and len(episodes[0][0]) == 2


class TestValidation:
    def test_small_classes_are_left_out(self, toy_dataset):
        ds = toy_dataset([0] * 5 + [1] * 5 + [2] * 5 + [3] * 2)
        wrapper = FewShotWrapper(ds, n_way=3, k_shot=2, q_query=2)
        assert wrapper.episode_classes == [0, 1, 2]
        with pytest.raises(ValueError, match="only 3 of the dataset's 4 classes"):
            FewShotWrapper(ds, n_way=4, k_shot=2, q_query=2)

    @pytest.mark.parametrize("kwargs", [{"n_way": 0}, {"k_shot": 0}, {"q_query": -1}, {"n_episodes": 0}])
    def test_invalid_sizes(self, five_classes, kwargs):
        with pytest.raises(ValueError):
            FewShotWrapper(five_classes, **kwargs)
