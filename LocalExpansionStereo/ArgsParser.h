#pragma once
#include <vector>
#include <map>
#include <string>

//reference https://stackoverflow.com/questions/3052579/explicit-specialization-in-non-namespace-scope
template<typename T>
struct identity { typedef T type; };

class ArgsParser
{
	std::vector<std::string> args;
	std::map<std::string, std::string> argMap;

	void parseArgments()
	{
		for (int i = 0; i < args.size(); i++)
		{
			if (args[i][0] == '-')
			{
				std::string name(&args[i][1]);
				if (i + 1 < args.size())
				{
					argMap[name] = args[i + 1];
					i++;
					std::cout << name << ": " << argMap[name] << std::endl;
				}
			}
		}
	}
	
	// I do not know
	/*template <typename T>
	T convertStringToValue(std::string str, identity<T>) const
	{
		//return (T)std::stod(str);
	}*/

	double convertStringToValue(std::string str, identity<double>) const{ return std::stod(str); }
	float convertStringToValue(std::string str, identity<float>) const{ return std::stof(str); }
	int convertStringToValue(std::string str, identity<int>) const{ return std::stoi(str); }
	std::string convertStringToValue(std::string str, identity<std::string>)const { return str; }
	bool convertStringToValue(std::string str, identity<bool>) const
	{
		if (str == "true") return true;
		if (str == "false") return false;
		return convertStringToValue(str, identity<int>()) != 0;
	}

public:
	ArgsParser(){}
	ArgsParser(int argn, const char **args)
	{
		for (int i = 0; i < argn; i++) {
			this->args.push_back(args[i]);
		}
		parseArgments();
	}
	ArgsParser(const std::vector<std::string>& args)
	{
		this->args = args;
		parseArgments();
	}

	template <typename T>
	bool TryGetArgment(std::string argName, T& value) const
	{
		auto it = argMap.find(argName);
		if (it == argMap.end())
			return false;

		value = convertStringToValue(it->second, identity<T>());
		return true;
	}
};
