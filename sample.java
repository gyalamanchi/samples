import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class RestServiceExample {

    public static void main(String[] args) {
        // Create an ExecutorService with a fixed thread pool
        ExecutorService executorService = Executors.newFixedThreadPool(5);

        // Create a Callable to make the REST call
        Callable<String> restServiceCallable = () -> {
            // Replace the URL with your actual REST service endpoint
            String restServiceUrl = "https://api.example.com/data";

            // Create an HttpClient
            HttpClient httpClient = HttpClient.newHttpClient();

            // Create an HttpRequest to the REST service
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(restServiceUrl))
                    .GET()
                    .build();

            // Make the REST call and retrieve the response
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

            // Return the response body
            return response.body();
        };

        try {
            // Submit the Callable to the ExecutorService
            Future<String> future = executorService.submit(restServiceCallable);

            // Wait for the result
            String result = future.get();

            // Print the result
            System.out.println("REST Service Response: " + result);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            // Shutdown the ExecutorService
            executorService.shutdown();
        }
    }
}
