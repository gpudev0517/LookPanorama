
            WorkerScript.onMessage = function(message) {
                //Calculate result (may take a while, using a naive algorithm)
                //var calculatedResult = triangle(message.row, message.column);
                //Send result back to main thread

                var message_ = message.temp
                WorkerScript.sendMessage({result: message_})
            }
