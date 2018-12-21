import numpy as np

class LocalNorm():
    """
    To normalize the array with local max and min value
    """
    def __init__(self, num=3):
        """
        :param num: the number of each axis will be divided.
        """
        self.num = num

    def one_to_blocks(self, data, norm=True):
        """

        :param data: np.array, data to get block
        :param norm: bool, normalize the block or not
        :return: data_blocked
        """
        self.y_interval = np.int(data.shape[0]/self.num)
        self.x_interval = np.int(data.shape[1]/self.num)
        self.data_blocked = np.zeros((self.num*self.num, self.y_interval, self.x_interval))
        print (data.shape)
        print (self.y_interval)
        print (self.x_interval)
        for i in range(self.num):
            for j in range(self.num):
                print (i)
                print (j)
                block = data[j*self.y_interval:(j+1)*self.y_interval, i*self.x_interval:(i+1)*self.x_interval]
                if norm == True:
                    block = self.normalizer(block)
                self.data_blocked[i*self.num + j] = block
        return self.data_blocked

    def normalizer(self, data, min_value=0, max_value=255):
        """
        :param data: np.array, data for normalizing
        :param min_value: after normalizing, the minimum value
        :param max_value: after normalizing, the maximum value
        :return: data_norm, data after normalizing
        """
        max_ = np.max(data)
        min_ = np.min(data)
        self.data_norm = max_value * (data - min_)/(max_ - min_) + min_value
        return self.data_norm

    def blocks_to_one(self, block_data):
        """
        :param block_data: np.array, shape=[self.num*self.num, self.y_interval, self.x_interval]
                            the blocked data
        :return: col_block, np.array, shape=[self.num*self.y_interval, self.num*self.x_interval]
                            two dimension data, combined by blocked data
        """
        for i in range(self.num):
            for j in range(self.num):
                if j == 0:
                    row_block = block_data[i*self.num + j]
                else:
                    row_block = np.concatenate((row_block, block_data[i*self.num + j]), axis=0)
            if i == 0:
                self.col_block = row_block
            else:
                self.col_block = np.concatenate((self.col_block, row_block), axis=1)
        return self.col_block

    def get_params(self):
        params = {'blocked data' : self.data_blocked, 'data norm' : self.data_norm,\
                 'recover data' : self.col_block}
        return params

    def run(self, data):
        """
        Run the normalize data by blocks.
        :param data: np.array. raw data
        :return: recover_data, np.array
            normalized by blocks result
        """
        blocked_data = self.one_to_blocks(data)
        recover_data = self.blocks_to_one(blocked_data)
        return recover_data

