import typing as tp

from dora import Explorer
import treetable as tt


class SimpleExplorer(Explorer):
    test_metrics: tp.List[str] = ['pesq', 'stoi']

    def get_grid_metrics(self):
        return [
            tt.group("train", [
                tt.leaf("epoch"),
                tt.leaf("train", ".3f"),
                tt.leaf("valid", ".3f"),
                tt.leaf("best", ".3f"),
            ], align=">"),
            tt.group("test", [
                tt.leaf("pesq", ".3f"),
                tt.leaf("stoi", ".3f"),
            ], align=">")
        ]

    def process_history(self, history: tp.List[dict]) -> dict:
        """Process the history, typically loaded from the
        `history.json` file as a list of dict, one entry per epoch.
        You get a chance to reorganize stuff here, or maybe perform
        some extra processing, and should return a single dict,
        potentially with nested dict.
        """
        train = {
            'epoch': len(history),
        }
        test = {}
        if history:
            metrics = history[-1]
            train.update({
                'train': metrics['train'],
                'valid': metrics['valid'],
                'best': metrics['best']})

            for name in self.test_metrics:
                if name in metrics:
                    test[name] = metrics[name]

        return {"train": train, "test": test}
