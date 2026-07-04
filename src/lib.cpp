#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

class Animal {
public:
  std::string name;
  int age;

  Animal(const std::string &name, int age) : name(name), age(age) {}

  std::string speak() const { return name + " says hello!"; }
  int test() const { return 10; }
};

PYBIND11_MODULE(goudacpp, m) {
  m.doc() = "Animal module";

  py::class_<Animal>(m, "Animal")
      .def(py::init<const std::string &, int>())
      .def("speak", &Animal::speak)
      .def("testt", &Animal::test)
      .def_readwrite("name", &Animal::name)
      .def_readwrite("age", &Animal::age);
}
