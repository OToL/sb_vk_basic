#include "utility.h"
#include <sb_core/conversion.h>

#include <sb_std/algorithm>

#include <cmath>

sb::u32 sb::getMipLevelCount(int width, int height)
{
    return numericConv<u32>(std::floor(std::log2(sbstd::max(width, height))) + 1);
}

