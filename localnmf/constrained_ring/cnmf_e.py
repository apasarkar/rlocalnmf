import torch
import logging
import math


class RingModel:

    def __init__(self, d1: int, d2: int, radius: int, device: str='cpu',
                 order: str="F"):
        """
        Ring Model object manages the state of the ring model during the model fit phase.

        Args:
            d1 (int): the 0th dimension of the FOV (in python indexing)
            d2 (int): the 1st dimension of the FOV (in python indexing)
            radius (int): the ring radius
            device (str): which device the pytorch data lies on
            order (str): the order used to reshape from 1D to 2D (and vice versa)
        """
        self._shape = (d1, d2)
        self._radius = radius
        self._device = device
        self._order = order
        self.weights = torch.ones((d1 * d2), device=device)
        self.support = torch.ones((self.shape[0]*self.shape[1]), device=self.device, dtype=torch.float32)

        self._dim1_ring, self._dim2_ring, self._ring_indices = self._precompute_ring_info()

    @property
    def shape(self):
        return self._shape

    @property
    def order(self):
        return self._order

    @property
    def device(self):
        return self._device

    @property
    def radius(self):
        return self._radius

    @property
    def weights(self):
        """
        The ring model uses a constant weight assumption: that pixel "i" of the data can be explained as a scaled
        average of the pixels in a ring surrounding pixel "i". This is enforced by a diagonal weight matrix: d_weight.

        Returns:
            d_weights (torch.sparse_coo_tensor): (d1*d2, d1*d2) diagonal matrix
        """
        return self._weights

    @weights.setter
    def weights(self, new_weights):
        """
        Sets the weights
        Args:
            new_weights (torch.tensor): Shape (d1*d2)
        """
        net_pixels = (self.shape[0] * self.shape[1])
        index_values = torch.arange(net_pixels, device=self.device, dtype=torch.long)
        self._weights = torch.sparse_coo_tensor(torch.stack([index_values, index_values]), new_weights,
                                             (net_pixels, net_pixels)).coalesce()


    @property
    def support(self):
        """
        The ring model only operates on pixels that do not contain spatial footprints. This is enforced by a diagonal
        mask matrix, D_{mask}. The i-th entry  is 0 if pixel i contains neural footprints, otherwise it is 1

        Returns:
            d_mask (torch.sparse_coo_tensor): (d1*d2, d1*d2) diagonal matrix represe
        """
        return self._support

    @support.setter
    def support(self, new_mask):
        """
        Args:
            new_mask (torch.tensor): Shape (d1*d2), index i is 0 if pixel i contains neural signal, otherwise it is 0
        """
        net_pixels = (self.shape[0]*self.shape[1])
        index_values = torch.arange(net_pixels, device=self.device, dtype=torch.long)
        self._support = torch.sparse_coo_tensor(torch.stack([index_values, index_values]), new_mask,
                                                (net_pixels, net_pixels)).coalesce()
    def _precompute_ring_info(self):
        d1, d2 = self.shape
        dim1_spread = torch.arange(-(self.radius + 1), (self.radius + 2), device=self.device)
        dim2_spread = torch.arange(-(self.radius + 1), (self.radius + 2), device=self.device)
        spr1, spr2 = torch.meshgrid([dim1_spread, dim2_spread], indexing='ij')
        norms = torch.sqrt(spr1 * spr1 + spr2 * spr2)
        outputs = torch.logical_and(norms >= self.radius, norms < self.radius + 1).to(self.device)

        dim1_ring = spr1[outputs].flatten().squeeze().long()
        dim2_ring = spr2[outputs].flatten().squeeze().long()

        # flatten the 2D ring representation to 1D for efficiency
        if self.order == "C":
            ring_indices = dim1_ring * d2 + dim2_ring
        elif self.order == "F":
            ring_indices = dim1_ring + d1 * dim2_ring
        else:
            raise ValueError("Not a valid ordering")

        return dim1_ring, dim2_ring, ring_indices

    def construct_ring_matrix(self, rows_to_generate):
        d1, d2 = self.shape
        #Define the "column indices" of the (d1*d2, d1*d2) sparse matrix (every row is a ring) in 1D representation.
        column_indices = rows_to_generate.unsqueeze(1) + self._ring_indices.unsqueeze(0)
        row_indices = rows_to_generate.unsqueeze(1) + torch.zeros((1, self._ring_indices.shape[0]),
                                                                  device=self.device, dtype=torch.long)

        #preliminary filter
        good_components = torch.logical_and(column_indices >= 0, column_indices < d1 * d2)
        if self.order == "C":
            '''
            We get rid of values that are out of bounds
            '''
            twod_column_indices = (rows_to_generate % d2).unsqueeze(1) + self._dim2_ring.unsqueeze(0)
            good_components = torch.logical_and(good_components, twod_column_indices >= 0)
            good_components = torch.logical_and(good_components, twod_column_indices < d2)

            twod_row_indices = (rows_to_generate // d2).unsqueeze(1) + self._dim1_ring.unsqueeze(0)
            good_components = torch.logical_and(good_components, twod_row_indices >= 0)
            good_components = torch.logical_and(good_components, twod_row_indices < d1)
        elif self.order == "F": #Make sure the vertical indices do not shift by columns
            '''
            good_components, as defined above, will filter out columns that are out of bounds, now we filter out rows
            '''
            twod_column_indices = (rows_to_generate // d1).unsqueeze(1) + self._dim2_ring.unsqueeze(0)
            good_components = torch.logical_and(good_components, twod_column_indices >= 0)
            good_components = torch.logical_and(good_components, twod_column_indices < d2)

            twod_row_indices = (rows_to_generate % d1).unsqueeze(1) + self._dim1_ring.unsqueeze(0)
            good_components = torch.logical_and(good_components, twod_row_indices >= 0)
            good_components = torch.logical_and(good_components, twod_row_indices < d1)

        column_indices = column_indices[good_components]
        row_indices = row_indices[good_components]
        values = torch.ones(row_indices.shape, device=self.device, dtype=torch.float32)

        return  torch.sparse_coo_tensor(torch.stack([row_indices, column_indices]), values,
                                        (rows_to_generate.shape[0], self.shape[0]*self.shape[1])).coalesce()

    def zero_weights(self):
        self.weights = torch.zeros((self.shape[0]*self.shape[1]), device=self.device)

    def reset_weights(self):
        self.weights = torch.ones((self.shape[0]*self.shape[1]), device=self.device)