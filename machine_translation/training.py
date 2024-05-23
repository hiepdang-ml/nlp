import os
from typing import List, Dict, Tuple, Optional, Literal, Callable
import datetime as dt

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch.nn as nn
from torchtext.data.metrics import bleu_score

from machine_translation.datasets import EnglishFrenchDataset
from machine_translation.seq2seq import Seq2Seq

from utils import Accumulator, EarlyStopping, Timer, Logger, CheckPointSaver


class Trainer:

    def __init__(
        self, 
        model: Seq2Seq, 
        dataset: EnglishFrenchDataset,
        split_ratios: Tuple[int, int, int] = (0.8, 0.1, 0.1),
        train_batch_size: int = 16,
        val_batch_size: int = 16,
        test_batch_size: int = 16,
        learning_rate: float = 0.005,
        checkpoint_output: Optional[str] = f'{os.environ["PYTHONPATH"]}/.checkpoint/{dt.datetime.now().strftime("%Y%m%d%H%M%S")}',
    ) -> None:
        self.model: Seq2Seq = model
        self.dataset: EnglishFrenchDataset = dataset
        self.train_batch_size: int = train_batch_size
        self.val_batch_size: int = val_batch_size
        self.test_batch_size: int = test_batch_size
        self.learning_rate: float = learning_rate
        self.checkpoint_output: str = checkpoint_output

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset=dataset, lengths=split_ratios)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=train_batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=val_batch_size, shuffle=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=test_batch_size, shuffle=False)

        self._target_pad_id: int = dataset.target_vocab.find_ids(['<pad>']).pop()
        self.optimizer: torch.optim.Optimizer =  torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.train_dataloader: DataLoader = DataLoader(
            dataset=self.train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
        )
        self.n_train_batches: int = len(self.train_dataloader)
        self.val_dataloader: DataLoader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
        )
        self.n_val_batches: int = len(self.val_dataloader)

    def loss_func(self, pred_tensor: torch.Tensor, gt_tensor: torch.Tensor) -> torch.Tensor:
        assert pred_tensor.dim() == 3
        assert gt_tensor.dim() == 2
        assert pred_tensor.shape[:2] == gt_tensor.shape     # (batch_size, n_steps)
        flattened_predictions: torch.Tesnor = pred_tensor.reshape(-1, pred_tensor.shape[-1])
        flattened_labels: torch.Tesnor = gt_tensor.reshape(-1)
        func: Callable = nn.CrossEntropyLoss(ignore_index=self._target_pad_id)
        loss: torch.Tensor = func(input=flattened_predictions, target=flattened_labels)
        return loss

    def train(self, n_epochs: int, patience: int, tolerance: float) -> None:
        train_metrics: Accumulator = Accumulator()
        early_stopping: EarlyStopping = EarlyStopping(patience=patience, tolerance=tolerance)
        timer: Timer = Timer()
        logger: Logger = Logger()
        check_point_saver: CheckPointSaver = CheckPointSaver(dirpath=self.checkpoint_output)

        # loop through each epoch
        epoch: int
        for epoch in range(1, n_epochs + 1):
            timer.start_epoch(epoch)
            self.model.train()
            # Loop through each batch
            step: int
            batch: Dict[Literal['source', 'target'], Tuple[torch.Tensor, int]]
            for step, batch in enumerate(self.train_dataloader, start=1):    # (N, 3, 256, 256), (N, 1, 5)
                # Start batch
                timer.start_batch(epoch=epoch, batch=step)
                # Get train data
                source_tensor: torch.Tensor = batch['source'][0]
                target_tensor: torch.Tensor = batch['target'][0]
                # Reset gradients
                self.optimizer.zero_grad()
                # Forward pass
                output: torch.Tensor = self.model(enc_x=source_tensor, dec_x=target_tensor[:, :-1])[0]
                # Compute loss
                loss: torch.Tensor = self.loss_func(pred_tensor=output, gt_tensor=target_tensor[:, 1:])
                # Backpropagation
                loss.backward()
                # Update weights
                self.optimizer.step()
                # Accumulate the metrics
                train_metrics.add(train_loss=loss.item())
                # End batch
                timer.end_batch(epoch=epoch, batch=step)
                # Log batch
                logger.log(
                    epoch=epoch, n_epochs=n_epochs, 
                    batch=step, n_batches=self.n_train_batches, 
                    took=timer.time_batch(epoch=epoch, batch=step), 
                    train_loss=train_metrics['train_loss'] / step, 
                )

            # Save checkpoint
            check_point_saver.save(model=self.model, filename=f'epoch{epoch}.pt')
            # Evaluate
            val_loss: float
            val_bleu_score: float
            val_loss, val_bleu_score = self.evaluate(dataloader=self.val_dataloader)
            # End epoch
            timer.end_epoch(epoch)
            # Log epoch
            logger.log(
                epoch=epoch, n_epochs=n_epochs, 
                took=timer.time_epoch(epoch=epoch),
                train_loss=train_metrics['train_loss'] / self.n_train_batches,
                val_loss=val_loss,
                val_bleu_score=val_bleu_score,
            )
            # Reset metric records for next epoch
            train_metrics.reset()
            # Early stopping
            early_stopping(value=val_bleu_score)
            if early_stopping:
                print('Early Stopped')
                break

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        val_metrics: Accumulator = Accumulator()
        with torch.no_grad():
            batch: Dict[Literal['source', 'target'], Tuple[torch.Tensor, int]]
            for batch in dataloader:
                source_tensor: torch.Tensor = batch['source'][0]
                target_tensor: torch.Tensor = batch['target'][0]
                # Forward pass
                output: torch.Tensor = self.model(enc_x=source_tensor, dec_x=target_tensor)[0]
                # Get predicted translations
                best_token_ids: torch.Tensor = output.argmax(dim=2) # shape = (batch_size, n_steps)
                pred_tokens: List[List[int]] = [self.dataset.target_vocab.find_tokens(ids=sample) for sample in best_token_ids.tolist()]
                # Get grouth-truth translations
                gt_tokens = [[self.dataset.target_vocab.find_tokens(ids=sample)] for sample in target_tensor.tolist()]
                # Compute BLEU score
                score: float = bleu_score(candidate_corpus=pred_tokens, references_corpus=gt_tokens, max_n=4)
                val_metrics.add(bleu_score=score)
                # Compute loss
                loss: float = self.loss_func(pred_tensor=output, gt_tensor=target_tensor).item()
                val_metrics.add(loss=loss)

        mean_bleu_score: float = val_metrics['bleu_score'] / len(self.val_dataloader)
        mean_loss: float = val_metrics['loss'] / len(self.val_dataloader)
        return mean_loss, mean_bleu_score




        
    
