from mxnet import nd


def nms(hm):
    hm_max = nd.Pooling(data=hm, kernel=(3, 3), pool_type='max',
                        stride=(1, 1), pad=(1, 1))
    max_idx = (hm_max == hm)
    hm = nd.broadcast_mul(hm, max_idx)
    return hm


def topk(hm, k=100):

    ctx = hm.context
    
    batch_size, cat, height, width = hm.shape

    hm = nms(hm)
    hm = nd.reshape(hm, (0, 0, -1))
    topk_scores, topk_idx = nd.topk(hm, k=k, ret_typ='both')
    
    topk_x_idx = nd.floor(topk_idx/width)
    topk_x_idx = nd.reshape(topk_x_idx, (0, -1))
    
    topk_y_idx = (topk_idx%height)
    topk_y_idx = nd.reshape(topk_y_idx, (0, -1))
    
    topk_scores = nd.reshape(topk_scores, (0, -1))
    topk_cat_scores, topk_cat_idx = nd.topk(topk_scores, k=k, ret_typ='both')
    
    cls_id = nd.floor(topk_cat_idx/k)
    
    batch_idx = nd.repeat(nd.arange(batch_size), repeats=k).reshape((1, -1))
    batch_idx = batch_idx.as_in_context(ctx)
    topk_cat_idx = nd.reshape(topk_cat_idx, (1, -1))
    topk_cat_idices = nd.concat(batch_idx, topk_cat_idx, dim=0)

    topk_cat_x_idx = nd.gather_nd(topk_x_idx, topk_cat_idices)
    topk_cat_x_idx = nd.reshape(topk_cat_x_idx, (batch_size, k))

    topk_cat_y_idx = nd.gather_nd(topk_y_idx, topk_cat_idices)
    topk_cat_y_idx = nd.reshape(topk_cat_y_idx, (batch_size, k))
    
    return topk_cat_x_idx, topk_cat_y_idx, cls_id


def get_pred_result(hm_pred, offset_pred, wh_pred, k=100):
    ctx = hm_pred.context
    batch_size, num_classes, _, _ = hm_pred.shape
    topk_cat_x_idx, topk_cat_y_idx, cls_id = topk(hm_pred, k=k)
    
    batch_index = nd.arange(batch_size)
    batch_indices = nd.repeat(batch_index, repeats=num_classes)
    batch_indices = nd.reshape(batch_indices, (1, batch_size*k))
    batch_indices = batch_indices.as_in_context(ctx)
    
    cls_id = nd.reshape(cls_id, (1, batch_size*k))
    topk_cat_y_idx = nd.reshape(topk_cat_y_idx, (1, batch_size*k))
    topk_cat_x_idx = nd.reshape(topk_cat_x_idx, (1, batch_size*k))
    
    score_indices = nd.concat(batch_indices, cls_id, topk_cat_y_idx, topk_cat_x_idx, dim=0)
    
    scores = nd.gather_nd(hm_pred, score_indices)
    
    fake_idx_0 = nd.zeros_like(nd.arange(batch_size*k)).reshape((1, -1))
    fake_idx_0 = fake_idx_0.as_in_context(ctx)
    fake_idx_1 = nd.ones((1, batch_size*k))
    fake_idx_1 = fake_idx_1.as_in_context(ctx)

    fake_indices_0 = nd.concat(batch_indices, fake_idx_0, topk_cat_y_idx, topk_cat_x_idx, dim=0)
    fake_indices_1 = nd.concat(batch_indices, fake_idx_1, topk_cat_y_idx, topk_cat_x_idx, dim=0)
    x_offset = nd.gather_nd(offset_pred, fake_indices_0)
    y_offset = nd.gather_nd(offset_pred, fake_indices_1)

    h = nd.gather_nd(wh_pred, fake_indices_0)
    w = nd.gather_nd(wh_pred, fake_indices_1)

    x_offset_ = nd.broadcast_mul(topk_cat_x_idx, x_offset)
    y_offset_ = nd.broadcast_mul(topk_cat_y_idx, y_offset)

    topk_cat_x_idx = nd.broadcast_add(topk_cat_x_idx, x_offset_)
    topk_cat_y_idx = nd.broadcast_add(topk_cat_y_idx, y_offset_)

    xmin = topk_cat_x_idx - w/2
    ymin = topk_cat_y_idx - h/2
    xmax = topk_cat_x_idx + w/2
    ymax = topk_cat_y_idx + h/2
    
    xmin = nd.reshape(xmin, (batch_size, k)).expand_dims(axis=-1)
    ymin = nd.reshape(ymin, (batch_size, k)).expand_dims(axis=-1)
    xmax = nd.reshape(xmax, (batch_size, k)).expand_dims(axis=-1)
    ymax = nd.reshape(ymax, (batch_size, k)).expand_dims(axis=-1)
    cls_id = nd.reshape(cls_id, (batch_size, k)).expand_dims(axis=-1)
    scores = nd.reshape(scores, (batch_size, k)).expand_dims(axis=-1)

    results = nd.concat(xmin, ymin, xmax, ymax, cls_id, scores, dim=-1)

    return results

