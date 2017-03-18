#include <iostream>

class prefixbuf : public std::streambuf
{
	std::string prefix;
	std::streambuf* sbuf;
	bool need_prefix;

	int sync()
	{
		return this->sbuf->pubsync();
	}

	int overflow(int c)
	{
		if (c != std::char_traits<char>::eof())
		{
			if (this->need_prefix && !this->prefix.empty() &&
				this->prefix.size() != this->sbuf->sputn(&this->prefix[0], this->prefix.size()))
			{
				return std::char_traits<char>::eof();
			}
			this->need_prefix = (c == '\n');
		}
		return this->sbuf->sputc(c);
	}

public :

	const std::string& getPrefix() { return prefix; }
	
	void setPrefix(const std::string& prefix_) { prefix = prefix_; }

	prefixbuf(std::streambuf* sbuf, std::string const& prefix_ = "") :
	
		sbuf(sbuf), prefix(prefix_), need_prefix(true)
	
	{ }
};

class oprefixstream : private virtual prefixbuf, public std::ostream
{
public:

	const std::string& getPrefix() { return prefixbuf::getPrefix(); }
	
	void setPrefix(const std::string& prefix_) { prefixbuf::setPrefix(prefix_); }

	oprefixstream(std::ostream& out, std::string const& prefix = "") :
	
		prefixbuf(out.rdbuf(), prefix),
		std::ios(static_cast<std::streambuf*>(this)),
		std::ostream(static_cast<std::streambuf*>(this))
		
	{ }
};

