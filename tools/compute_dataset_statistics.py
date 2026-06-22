"""计算 Transformer 训练所需的特征归一化统计量。

加载未归一化的训练集，聚合卫星和任务特征的均值/标准差，将结果保存到
模型/数据集代码使用的 ``STATISTICS_PATH``。
"""

from constellation import MAX_TIME_STEP, DATA_ROOT
from constellation import STATISTICS_PATH
from constellation.new_transformers.dataset import Dataset, Batch, Statistics
from todd.utils import Statistician
from tqdm import tqdm
import einops
import torch

CHUNK_SIZE = 10_000


def main() -> None:
    dataset = Dataset(split='train', batch_size=MAX_TIME_STEP, normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset, None, num_workers=32)
    constellation_statistician = Statistician(CHUNK_SIZE)
    taskset_statistician = Statistician(CHUNK_SIZE)
    batch: Batch
    for batch in tqdm(dataloader):
        constellation_statistician.update(
            einops.rearrange(
                batch.constellation_data,
                't ns d -> (t ns) d',
            ),
        )
        taskset_statistician.update(
            einops.rearrange(
                batch.tasks_data,
                't nt d -> (t nt) d',
            ),
        )
    statistics = Statistics(
        constellation_statistician.compute_mean(),
        constellation_statistician.compute_variance().sqrt(),
        taskset_statistician.compute_mean(),
        taskset_statistician.compute_variance().sqrt(),
    )
    torch.save(statistics, STATISTICS_PATH)


if __name__ == '__main__':
    main()
