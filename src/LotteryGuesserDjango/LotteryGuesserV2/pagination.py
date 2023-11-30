from django.core.paginator import EmptyPage
from rest_framework import pagination
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response


class LargeResultsSetPagination(PageNumberPagination):
    page_size = 25
    page_size_query_param = 'page_size'
    max_page_size = 10000


class StandardResultsSetPagination(PageNumberPagination):
    page_size = 100
    page_size_query_param = 'page_size'
    max_page_size = 1000


class CustomPagination(pagination.PageNumberPagination):
    page_size = 10
    page_size_query_param = 'page_size'
    max_page_size = 100000
    page_query_param = 'page'

    def paginate_queryset(self, queryset, request, view=None):
        self.page_size = self.get_page_size(request)
        if not self.page_size:
            return None

        paginator = self.django_paginator_class(queryset, self.page_size)
        page_number = request.query_params.get(self.page_query_param, 1)
        if page_number in self.last_page_strings:
            page_number = paginator.num_pages


        try:

            self.page = paginator.page(page_number)
            return list(self.page)
        except Exception as exc:
            self.page = paginator.page(1)
            # Here it is
            return []

    def get_paginated_response(self, data):
        response = Response({
            'page_size': self.page_size,
            'total_objects': self.page.paginator.count,
            'total_pages': self.page.paginator.num_pages,
            'current_page_number': self.page.number,
            'results': data,
        })
        return response

    def get_paginated_response_schema(self, schema):
        return {
            'type': 'object',
            'properties': {
                'page_size': {
                    'type': 'integer',
                    'example': 123,
                },
                'total_pages': {
                    'type': 'integer',
                    'example': 123,
                },
                'current_page_number': {
                    'type': 'integer',
                    'example': 123,
                },
                'total_objects': {
                    'type': 'integer',
                    'example': 123,
                },
                'results': schema,
            },
        }


class SiteCustomPagination(pagination.PageNumberPagination):
    page_size = 10
    page_size_query_param = 'page_size'
    max_page_size = 100000
    page_query_param = 'page'
    site = None

    def paginate_queryset(self, queryset, request, view=None, site=None):
        self.page_size = self.get_page_size(request)
        if not self.page_size:
            return None

        paginator = self.django_paginator_class(queryset, self.page_size)
        page_number = request.query_params.get(self.page_query_param, 1)
        if page_number in self.last_page_strings:
            page_number = paginator.num_pages

        self.site = site



        try:

            self.page = paginator.page(page_number)
            return list(self.page)
        except Exception as exc:
            self.page = paginator.page(1)
            # Here it is
            return []

    def get_paginated_response(self, data):
        response = Response({
            'page_size': self.page_size,
            'total_objects': self.page.paginator.count,
            'total_pages': self.page.paginator.num_pages,
            'current_page_number': self.page.number,
            'site': self.site,
            'results': data,
        })
        return response

    def get_paginated_response_schema(self, schema):
        return {
            'type': 'object',
            'properties': {
                'page_size': {
                    'type': 'integer',
                    'example': 123,
                },
                'total_pages': {
                    'type': 'integer',
                    'example': 123,
                },
                'current_page_number': {
                    'type': 'integer',
                    'example': 123,
                },
                'total_objects': {
                    'type': 'integer',
                    'example': 123,
                },
                'site': {
                    'id': {
                        'type': 'integer',
                        'example': 123,
                    },
                    'name': {
                        'type': 'string',
                        'example': 'example',
                    },
                    'pages' : {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'id': {
                                    'type': 'integer',
                                    'example': 123,
                                },
                                'slug': {
                                    'type': 'string',
                                    'example': 'example',
                                },
                                'version': {
                                    'type': 'integer',
                                    'example': 123,
                                },
                            },
                        },
                    },
                },
                'results': schema,
            },
        }
