#ifndef _UTILITIES_H_
#define _UTILITIES_H_

#include <regex>

// use to perform split to obtain normal // 
std::vector<std::string> split(const std::string str, const std::string regex_str)
{   // a yet more concise form!
    return { std::sregex_token_iterator(str.begin(), str.end(), std::regex(regex_str), -1), std::sregex_token_iterator() };
}

#endif