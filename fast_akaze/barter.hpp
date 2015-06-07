#ifndef __BARTER_HPP__
#define __BARTER_HPP__

#include <memory>  // std::unique_ptr
#include <atomic>  // std::atomic


/*
  Description:
    A simple header-only class template to allow safe data exchange between threads.
 */
template <typename Typ_, typename Del_ = std::default_delete<Typ_> >
class barter {

    // The shared object: the only contact point between two threads
    std::atomic<Typ_ *> _pinned_object;

public:
    using Ptr = std::unique_ptr < Typ_, Del_ > ;

    barter()
    {
        _pinned_object = new Typ_{};
    }

    ~barter()
    {
        // call the destructor for the pinned_object
        Ptr().reset(_pinned_object);
    }

    void exchange(Ptr & p_)
    {
        Typ_ * oldptr = _pinned_object.exchange(p_.release());
        p_.reset(oldptr);
    }
};

#endif  // !__BARTER_HPP__
