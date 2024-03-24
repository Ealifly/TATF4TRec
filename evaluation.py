from recsys_metrics import hit_rate
from recsys_metrics import recall
from recsys_metrics import normalized_dcg
from recsys_metrics import mean_average_precision
from recsys_metrics import precision
from recsys_metrics import mean_reciprocal_rank


def evaluation(target, prediction):
    hr1 = hit_rate(prediction, target, k=1)
    # hr2 = hit_rate(prediction, target, k=2)
    hr3 = hit_rate(prediction, target, k=3)
    # hr4 = hit_rate(prediction, target, k=4)
    hr5 = hit_rate(prediction, target, k=5)
    # hr10 = hit_rate(prediction, target, k=10)
    # hr20 = hit_rate(prediction, target, k=20)
    # hr50 = hit_rate(prediction, target, k=50)
    # hr100 = hit_rate(prediction, target, k=100)

    # rec1 = recall(prediction, target, k=1)
    # rec5 = recall(prediction, target, k=5)
    # rec10 = recall(prediction, target, k=10)
    # rec20 = recall(prediction, target, k=20)
    # rec50 = recall(prediction, target, k=50)
    # rec100 = recall(prediction, target, k=100)

    # ndcg1 = normalized_dcg(prediction, target, k=1)
    ndcg3 = normalized_dcg(prediction, target, k=3)
    ndcg5 = normalized_dcg(prediction, target, k=5)
    # ndcg10 = normalized_dcg(prediction, target, k=10)
    # ndcg20 = normalized_dcg(prediction, target, k=20)
    # ndcg50 = normalized_dcg(prediction, target, k=50)
    # ndcg100 = normalized_dcg(prediction, target, k=100)

    # map1 = mean_average_precision(prediction, target, k=1)
    # map5 = mean_average_precision(prediction, target, k=5)
    # map10 = mean_average_precision(prediction, target, k=10)
    # map20 = mean_average_precision(prediction, target, k=20)
    # map50 = mean_average_precision(prediction, target, k=50)
    # map100 = mean_average_precision(prediction, target, k=100)

    # prec1 = precision(prediction, target, k=1)
    # prec3 = precision(prediction, target, k=5)
    # prec5 = precision(prediction, target, k=5)
    # prec10 = precision(prediction, target, k=10)
    # prec20 = precision(prediction, target, k=20)
    # prec50 = precision(prediction, target, k=50)
    # prec100 = precision(prediction, target, k=100)

    # mrr1 = mean_reciprocal_rank(prediction, target, k=1)
    # mrr5 = mean_reciprocal_rank(prediction, target, k=5)
    # mrr10 = mean_reciprocal_rank(prediction, target, k=10)
    # mrr20 = mean_reciprocal_rank(prediction, target, k=20)
    # mrr50 = mean_reciprocal_rank(prediction, target, k=50)
    # mrr100 = mean_reciprocal_rank(prediction, target, k=100)

    # return hr1, hr5, hr10, hr20, hr50, hr100, \
    #     rec1, rec5, rec10, rec20, rec50, rec100, \
    #     ndcg1, ndcg5, ndcg10, ndcg20, ndcg50, ndcg100, \
    #     map1, map5, map10, map20, map50, map100, \
    #     prec1, prec5, prec10, prec20, prec50, prec100, \
    #     mrr1, mrr5, mrr10, mrr20, mrr50, mrr100
    return hr1, hr3, hr5, ndcg3, ndcg5
