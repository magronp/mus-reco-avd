#!/usr/bin/env python
# -*- coding: utf-8 -*-

import bottleneck as bn
import numpy as np
from scipy import sparse

__author__ = 'Paul Magron -- IRIT, Université de Toulouse, CNRS, France'
__docformat__ = 'reStructuredText'


def normalized_dcg(train_data, heldout_data, U, V, batch_users=5000, vad_data=None):
    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(ndcg_binary_batch(train_data, heldout_data, U, V.T, user_idx, vad_data=vad_data))
    ndcg = np.hstack(res)
    ndcg_mean = np.nanmean(ndcg)
    ndcg_std = np.nanstd(ndcg)

    return ndcg_mean, ndcg_std


def normalized_dcg_out(heldout_data, pred_ratings, batch_users=5000):

    # Iterate over user batches
    res = list()
    for user_idx in user_idx_generator(heldout_data.shape[0], batch_users):
        heldout_data_batch = heldout_data[user_idx]
        pred_ratings_batch = pred_ratings[user_idx, :]
        res.append(ndcg_binary_batch_out(heldout_data_batch, pred_ratings_batch))

    # Get the mean and std NDCG over users
    ndcg = np.hstack(res)
    ndcg_mean = np.nanmean(ndcg)
    ndcg_std = np.nanstd(ndcg)

    return ndcg_mean, ndcg_std


def normalized_dcg_at_k(train_data, heldout_data, U, V, batch_users=5000, k=100, vad_data=None):

    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(ndcg_binary_at_k_batch(train_data, heldout_data, U, V.T, user_idx, k=k, vad_data=vad_data))
    ndcg = np.nanmean(np.hstack(res))

    return ndcg


# Generate of list of user indexes for each batch
def user_idx_generator(n_users, batch_users):
    for start in range(0, n_users, batch_users):
        end = min(n_users, start + batch_users)
        yield slice(start, end)


# Make the predictions (compute the product users * items)
def _make_prediction(train_data, Et, Eb, user_idx, batch_users, vad_data=None):

    n_songs = train_data.shape[1]

    # exclude examples from training and validation (if any)
    item_idx = np.zeros((batch_users, n_songs), dtype=bool)
    item_idx[train_data[user_idx].nonzero()] = True
    if vad_data is not None:
        item_idx[vad_data[user_idx].nonzero()] = True
    X_pred = Et[user_idx].dot(Eb)
    X_pred[item_idx] = -np.inf

    return X_pred


def ndcg_binary_batch_out(heldout_data, pred_ratings):

    # Get all ranks
    all_rank = np.argsort(np.argsort(-pred_ratings, axis=1), axis=1)

    # Build the discount template
    tp = 1. / np.log2(np.arange(2, heldout_data.shape[1] + 2))
    all_disc = tp[all_rank]

    # Get the ground truth ratings
    true_ratings = (heldout_data > 0).tocoo()

    # Get the NDCG
    disc = sparse.csr_matrix((all_disc[true_ratings.row, true_ratings.col], (true_ratings.row, true_ratings.col)),
                             shape=all_disc.shape)
    DCG = np.array(disc.sum(axis=1)).ravel()
    IDCG = np.array([tp[:n].sum()
                     for n in heldout_data.getnnz(axis=1)])

    return DCG / IDCG


def ndcg_binary_batch(train_data, heldout_data, Et, Eb, user_idx, vad_data=None):
    '''
    normalized discounted cumulative gain for binary relevance
    '''
    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, Et, Eb, user_idx, batch_users, vad_data=vad_data)

    all_rank = np.argsort(np.argsort(-X_pred, axis=1), axis=1)
    # build the discount template
    tp = 1. / np.log2(np.arange(2, train_data.shape[1] + 2))
    all_disc = tp[all_rank]

    X_true_binary = (heldout_data[user_idx] > 0).tocoo()
    disc = sparse.csr_matrix((all_disc[X_true_binary.row, X_true_binary.col],
                              (X_true_binary.row, X_true_binary.col)),
                             shape=all_disc.shape)

    DCG = np.array(disc.sum(axis=1)).ravel()

    IDCG = np.array([tp[:n].sum()
                     for n in heldout_data[user_idx].getnnz(axis=1)])
    return DCG / IDCG


def ndcg_binary_at_k_batch(train_data, heldout_data, Et, Eb, user_idx, k=100, vad_data=None):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, Et, Eb, user_idx, batch_users, vad_data=vad_data)
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    heldout_batch = heldout_data[user_idx]
    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG


# EOF
