#include <chrono>
#include <thread>

class SafeQueue {
public:
	SafeQueue() {
		firstElement = 0;
		lastElement = 0;
	}

	~SafeQueue() {
		firstElement = 0;
		lastElement = 0;
	}

	//voeg een element toe aan het einde van de rij
	void push(float f) {
		
		while (getSize() >= (queueSize-1))
			std::this_thread::sleep_for(std::chrono::microseconds(100));
		
		
		queue[lastElement] = f;
		++lastElement;
		if(lastElement >= queueSize)
			lastElement = 0;

	}

	//verwijder het eerste element uit de rij
	void pop() {
		while (getSize() <= 0)
			std::this_thread::sleep_for(std::chrono::microseconds(100));

		++firstElement;
		if(firstElement >= queueSize)
			firstElement = 0;
	

	}

	//geef het i-de element van de rij terug (synchronisatiefout tussen getSize() en ret = queue)
	float get(unsigned int i) {
		while(true) {
			if(i < getSize())
				break;
		}

		float ret = queue[(firstElement+i) % queueSize];


		return ret;
	}


	//geef het aantal elementen in de rij terug
	unsigned int getSize() {
		unsigned int tmp = lastElement;
		if(lastElement < firstElement)
			tmp += queueSize;
		tmp = tmp-firstElement;
		return tmp;
	}
	
	

private:
	//indices voor de queue
	unsigned int firstElement, lastElement;

	const static unsigned int queueSize = 2;
	float queue[queueSize];

};
