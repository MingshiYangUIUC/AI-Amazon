#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>  // PyTorch support
#include <vector>
#include <array>

namespace py = pybind11;

// Directions array: 8 possible directions (N, S, E, W, NE, NW, SE, SW)
std::array<std::array<int, 2>, 8> directions = {{{1, 0}, {-1, 0}, {0, 1}, {0, -1}, 
                                                 {1, 1}, {1, -1}, {-1, 1}, {-1, -1}}};

// C++ version of the select_action function accepting a tensor and an integer
std::vector<std::vector<int>> select_action(torch::Tensor b, int role = 0) {
    int bsize = b.size(0);
    
    // Find coordinates of the role
    std::vector<std::array<int, 2>> coords;
    for (int i = 0; i < bsize; ++i) {
        for (int j = 0; j < bsize; ++j) {
            if (b[i][j].item<int>() == role) {
                coords.push_back({i, j});
            }
        }
    }

    // List to store the final actions
    std::vector<std::vector<int>> actions;

    // Loop over each role coordinate
    for (const auto& crd : coords) {

        // Helper board for shooting (copying the tensor to b1)
        torch::Tensor b1 = b.clone();
        b1[crd[0]][crd[1]] = 0;

        // Get move places and shooting places
        std::vector<std::array<int, 2>> movetgt;

        // Collect valid move targets
        for (const auto& dir : directions) {
            std::array<int, 2> ncrd = crd;  // Copy the current coordinate
            ncrd[0] += dir[0];
            ncrd[1] += dir[1];

            while (ncrd[0] >= 0 && ncrd[1] >= 0 && ncrd[0] < bsize && ncrd[1] < bsize && b[ncrd[0]][ncrd[1]].item<int>() == 0) {
                movetgt.push_back(ncrd);  // Store the valid move target
                ncrd[0] += dir[0];
                ncrd[1] += dir[1];
            }
        }

        // Collect valid shooting targets for each move target
        for (const auto& mt : movetgt) {
            for (const auto& dir : directions) {
                std::array<int, 2> nmt = mt;  // Copy the move target
                nmt[0] += dir[0];
                nmt[1] += dir[1];

                while (nmt[0] >= 0 && nmt[1] >= 0 && nmt[0] < bsize && nmt[1] < bsize && b1[nmt[0]][nmt[1]].item<int>() == 0) {
                    // Append the action {crd, mt, nmt} in the format [crd_x, crd_y, mt_x, mt_y, nmt_x, nmt_y]
                    actions.push_back({crd[0], crd[1], mt[0], mt[1], nmt[0], nmt[1]});
                    nmt[0] += dir[0];
                    nmt[1] += dir[1];
                }
            }
        }
    }

    return actions;
}

PYBIND11_MODULE(action_module, m) {
    m.def("select_action_cpp", &select_action, "Select all valid actions for a given role on the board");
}
