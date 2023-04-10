// Generated by the gRPC C++ plugin.
// If you make any local change, they will be lost.
// source: service.proto
#ifndef GRPC_service_2eproto__INCLUDED
#define GRPC_service_2eproto__INCLUDED

#include "service.pb.h"

#include <functional>
#include <grpcpp/generic/async_generic_service.h>
#include <grpcpp/support/async_stream.h>
#include <grpcpp/support/async_unary_call.h>
#include <grpcpp/support/client_callback.h>
#include <grpcpp/client_context.h>
#include <grpcpp/completion_queue.h>
#include <grpcpp/support/message_allocator.h>
#include <grpcpp/support/method_handler.h>
#include <grpcpp/impl/proto_utils.h>
#include <grpcpp/impl/rpc_method.h>
#include <grpcpp/support/server_callback.h>
#include <grpcpp/impl/server_callback_handlers.h>
#include <grpcpp/server_context.h>
#include <grpcpp/impl/service_type.h>
#include <grpcpp/support/status.h>
#include <grpcpp/support/stub_options.h>
#include <grpcpp/support/sync_stream.h>

namespace za {

class SAD final {
 public:
  static constexpr char const* service_full_name() {
    return "za.SAD";
  }
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    virtual ::grpc::Status validate(::grpc::ClientContext* context, const ::za::AudioRequest& request, ::za::AnomalyReply* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::za::AnomalyReply>> Asyncvalidate(::grpc::ClientContext* context, const ::za::AudioRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::za::AnomalyReply>>(AsyncvalidateRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::za::AnomalyReply>> PrepareAsyncvalidate(::grpc::ClientContext* context, const ::za::AudioRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::za::AnomalyReply>>(PrepareAsyncvalidateRaw(context, request, cq));
    }
    class async_interface {
     public:
      virtual ~async_interface() {}
      virtual void validate(::grpc::ClientContext* context, const ::za::AudioRequest* request, ::za::AnomalyReply* response, std::function<void(::grpc::Status)>) = 0;
      virtual void validate(::grpc::ClientContext* context, const ::za::AudioRequest* request, ::za::AnomalyReply* response, ::grpc::ClientUnaryReactor* reactor) = 0;
    };
    typedef class async_interface experimental_async_interface;
    virtual class async_interface* async() { return nullptr; }
    class async_interface* experimental_async() { return async(); }
   private:
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::za::AnomalyReply>* AsyncvalidateRaw(::grpc::ClientContext* context, const ::za::AudioRequest& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::za::AnomalyReply>* PrepareAsyncvalidateRaw(::grpc::ClientContext* context, const ::za::AudioRequest& request, ::grpc::CompletionQueue* cq) = 0;
  };
  class Stub final : public StubInterface {
   public:
    Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options = ::grpc::StubOptions());
    ::grpc::Status validate(::grpc::ClientContext* context, const ::za::AudioRequest& request, ::za::AnomalyReply* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::za::AnomalyReply>> Asyncvalidate(::grpc::ClientContext* context, const ::za::AudioRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::za::AnomalyReply>>(AsyncvalidateRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::za::AnomalyReply>> PrepareAsyncvalidate(::grpc::ClientContext* context, const ::za::AudioRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::za::AnomalyReply>>(PrepareAsyncvalidateRaw(context, request, cq));
    }
    class async final :
      public StubInterface::async_interface {
     public:
      void validate(::grpc::ClientContext* context, const ::za::AudioRequest* request, ::za::AnomalyReply* response, std::function<void(::grpc::Status)>) override;
      void validate(::grpc::ClientContext* context, const ::za::AudioRequest* request, ::za::AnomalyReply* response, ::grpc::ClientUnaryReactor* reactor) override;
     private:
      friend class Stub;
      explicit async(Stub* stub): stub_(stub) { }
      Stub* stub() { return stub_; }
      Stub* stub_;
    };
    class async* async() override { return &async_stub_; }

   private:
    std::shared_ptr< ::grpc::ChannelInterface> channel_;
    class async async_stub_{this};
    ::grpc::ClientAsyncResponseReader< ::za::AnomalyReply>* AsyncvalidateRaw(::grpc::ClientContext* context, const ::za::AudioRequest& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::za::AnomalyReply>* PrepareAsyncvalidateRaw(::grpc::ClientContext* context, const ::za::AudioRequest& request, ::grpc::CompletionQueue* cq) override;
    const ::grpc::internal::RpcMethod rpcmethod_validate_;
  };
  static std::unique_ptr<Stub> NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options = ::grpc::StubOptions());

  class Service : public ::grpc::Service {
   public:
    Service();
    virtual ~Service();
    virtual ::grpc::Status validate(::grpc::ServerContext* context, const ::za::AudioRequest* request, ::za::AnomalyReply* response);
  };
  template <class BaseClass>
  class WithAsyncMethod_validate : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithAsyncMethod_validate() {
      ::grpc::Service::MarkMethodAsync(0);
    }
    ~WithAsyncMethod_validate() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status validate(::grpc::ServerContext* /*context*/, const ::za::AudioRequest* /*request*/, ::za::AnomalyReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void Requestvalidate(::grpc::ServerContext* context, ::za::AudioRequest* request, ::grpc::ServerAsyncResponseWriter< ::za::AnomalyReply>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  typedef WithAsyncMethod_validate<Service > AsyncService;
  template <class BaseClass>
  class WithCallbackMethod_validate : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithCallbackMethod_validate() {
      ::grpc::Service::MarkMethodCallback(0,
          new ::grpc::internal::CallbackUnaryHandler< ::za::AudioRequest, ::za::AnomalyReply>(
            [this](
                   ::grpc::CallbackServerContext* context, const ::za::AudioRequest* request, ::za::AnomalyReply* response) { return this->validate(context, request, response); }));}
    void SetMessageAllocatorFor_validate(
        ::grpc::MessageAllocator< ::za::AudioRequest, ::za::AnomalyReply>* allocator) {
      ::grpc::internal::MethodHandler* const handler = ::grpc::Service::GetHandler(0);
      static_cast<::grpc::internal::CallbackUnaryHandler< ::za::AudioRequest, ::za::AnomalyReply>*>(handler)
              ->SetMessageAllocator(allocator);
    }
    ~WithCallbackMethod_validate() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status validate(::grpc::ServerContext* /*context*/, const ::za::AudioRequest* /*request*/, ::za::AnomalyReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual ::grpc::ServerUnaryReactor* validate(
      ::grpc::CallbackServerContext* /*context*/, const ::za::AudioRequest* /*request*/, ::za::AnomalyReply* /*response*/)  { return nullptr; }
  };
  typedef WithCallbackMethod_validate<Service > CallbackService;
  typedef CallbackService ExperimentalCallbackService;
  template <class BaseClass>
  class WithGenericMethod_validate : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithGenericMethod_validate() {
      ::grpc::Service::MarkMethodGeneric(0);
    }
    ~WithGenericMethod_validate() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status validate(::grpc::ServerContext* /*context*/, const ::za::AudioRequest* /*request*/, ::za::AnomalyReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithRawMethod_validate : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithRawMethod_validate() {
      ::grpc::Service::MarkMethodRaw(0);
    }
    ~WithRawMethod_validate() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status validate(::grpc::ServerContext* /*context*/, const ::za::AudioRequest* /*request*/, ::za::AnomalyReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void Requestvalidate(::grpc::ServerContext* context, ::grpc::ByteBuffer* request, ::grpc::ServerAsyncResponseWriter< ::grpc::ByteBuffer>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithRawCallbackMethod_validate : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithRawCallbackMethod_validate() {
      ::grpc::Service::MarkMethodRawCallback(0,
          new ::grpc::internal::CallbackUnaryHandler< ::grpc::ByteBuffer, ::grpc::ByteBuffer>(
            [this](
                   ::grpc::CallbackServerContext* context, const ::grpc::ByteBuffer* request, ::grpc::ByteBuffer* response) { return this->validate(context, request, response); }));
    }
    ~WithRawCallbackMethod_validate() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status validate(::grpc::ServerContext* /*context*/, const ::za::AudioRequest* /*request*/, ::za::AnomalyReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual ::grpc::ServerUnaryReactor* validate(
      ::grpc::CallbackServerContext* /*context*/, const ::grpc::ByteBuffer* /*request*/, ::grpc::ByteBuffer* /*response*/)  { return nullptr; }
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_validate : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithStreamedUnaryMethod_validate() {
      ::grpc::Service::MarkMethodStreamed(0,
        new ::grpc::internal::StreamedUnaryHandler<
          ::za::AudioRequest, ::za::AnomalyReply>(
            [this](::grpc::ServerContext* context,
                   ::grpc::ServerUnaryStreamer<
                     ::za::AudioRequest, ::za::AnomalyReply>* streamer) {
                       return this->Streamedvalidate(context,
                         streamer);
                  }));
    }
    ~WithStreamedUnaryMethod_validate() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status validate(::grpc::ServerContext* /*context*/, const ::za::AudioRequest* /*request*/, ::za::AnomalyReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status Streamedvalidate(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< ::za::AudioRequest,::za::AnomalyReply>* server_unary_streamer) = 0;
  };
  typedef WithStreamedUnaryMethod_validate<Service > StreamedUnaryService;
  typedef Service SplitStreamedService;
  typedef WithStreamedUnaryMethod_validate<Service > StreamedService;
};

}  // namespace za


#endif  // GRPC_service_2eproto__INCLUDED
